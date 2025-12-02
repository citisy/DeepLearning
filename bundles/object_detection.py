import copy
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import Dataset

from data_parse import DataRegister
from data_parse.cv_data_parse.data_augmentation import pixel_perturbation, scale, geometry, channel, RandomApply, Apply, complex
from data_parse.cv_data_parse.datasets.base import DataVisualizer
from metrics import object_detection
from processor import Process, DataHooks, bundled, BaseImgDataset
from utils import configs, cv_utils, os_lib, torch_utils, log_utils


class OdDataset(BaseImgDataset):
    def process_one(self, idx):
        ret = copy.deepcopy(self.iter_data[idx])
        if isinstance(ret['image'], str):
            ret['image_path'] = ret['image']
            ret['image'] = cv2.imread(ret['image'])

        ret['ori_image'] = ret['image']
        ret['ori_bboxes'] = ret['bboxes']
        ret['ori_classes'] = ret['classes']
        ret['idx'] = idx

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret


class OdProcess(Process):
    n_classes: int
    in_ch: int = 3
    input_size: int

    def get_model_inputs(self, loop_inputs, train=True):
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
        images = torch.stack(images)

        # note that, if the images have the same shape, minmax after stack if possible
        # it can reduce about 20 seconds per epoch to voc dataset
        images = images / 255

        r = dict(x=images)
        if train:
            r.update(
                gt_boxes=[torch.from_numpy(ret['bboxes']).to(self.device) for ret in loop_inputs],
                gt_cls=[torch.from_numpy(ret['classes']).to(self.device) for ret in loop_inputs]
            )

        return r

    def on_train_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs, train=True)

        # note that, amp method can make the model run in dtype of half
        # even though input has dtype of torch.half and weight has dtype of torch.float
        # so that, it would run in lower memory and cost less time
        with torch.amp.autocast('cuda', enabled=True):
            output = self.model(**inputs)

        return output

    def metric(self, *args, **kwargs):
        process_results = self.predict(**kwargs)

        metric_results = {}
        for name, results in process_results.items():
            gt_rets = results['trues']
            det_rets = results['preds']
            df = object_detection.EasyMetric(verbose=False).quick_metric(gt_rets, det_rets, save_path=f'{self.work_dir}/result.csv')

            # avoid to log too much, only log the main model
            if name == self.model_name:
                log_info = {
                    # 'table': self.wandb.Table(dataframe=df)
                }
                for i in df.index:
                    cls_name = i
                    if hasattr(self, 'cls_alias'):
                        try:
                            cls_name = self.cls_alias[i]
                        except:
                            pass
                    log_info[f'metrics/{cls_name}'] = df['ap'][i]

                self.trace(log_info, bundled.WANDB)

            metric_results[name] = dict(
                per_class_result=df,
                score=df['ap']['mean']
            )

        return metric_results

    def on_val_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs, train=False)

        model_results = {}
        for name, model in self.models.items():
            outputs = model(**inputs)
            outputs = [{k: v.to('cpu').numpy() for k, v in t.items()} for t in outputs]

            preds = []
            for (output, ret) in zip(outputs, loop_inputs):
                output = configs.ConfigObjParse.merge_dict(ret, output)
                output = self.val_data_restore(output)
                preds.append(dict(
                    bboxes=output['bboxes'],
                    classes=output['classes'],
                    confs=output['confs']
                ))

            model_results[name] = dict(
                outputs=outputs,
                preds=preds,
            )

        return model_results

    def on_val_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        loop_inputs = loop_objs['loop_inputs']
        model_results = loop_objs['model_results']

        for name, results in model_results.items():
            r = process_results.setdefault(name, dict())
            trues = r.setdefault('trues', [])
            preds = r.setdefault('preds', [])
            for ret, pred in zip(loop_inputs, results['preds']):
                trues.append(dict(
                    _id=ret['_id'],
                    bboxes=ret['ori_bboxes'],
                    classes=ret['ori_classes'],
                ))
                pred['_id'] = ret['_id']
                preds.append(pred)

    def visualize(self, loop_objs, n, **kwargs):
        loop_inputs = loop_objs['loop_inputs']
        model_results = loop_objs['model_results']

        for name, results in model_results.items():
            vis_trues = []
            vis_preds = []
            for i in range(n):
                true = loop_inputs[i]
                pred = results['preds'][i]

                vis_trues.append(dict(
                    _id=true['_id'],
                    image=true['ori_image'],
                    bboxes=true['ori_bboxes'],
                    classes=true['ori_classes']
                ))

                pred['image'] = true['ori_image']
                vis_preds.append(pred)

            cls_alias = self.__dict__.get('cls_alias')
            cache_image = DataVisualizer(f'{self.cache_dir}/{loop_objs["epoch"]}/{name}', verbose=False, pbar=False)(
                vis_trues, vis_preds, return_image=True, cls_alias=cls_alias
            )
            self.get_log_trace(bundled.WANDB).setdefault(f'val_image/{name}', []).extend(
                [self.wandb.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=Path(r['_id']).stem) for img, r in zip(cache_image, vis_trues)]
            )

    def fragment_predict(self, image: np.ndarray, **kwargs):
        images, coors = cv_utils.fragment_image(image, size=self.input_size, over_ratio=0.5, overlap_ratio=0.2)
        results = self.batch_predict(images)

        bboxes = []
        classes = []
        confs = []

        for (x1, y1, x2, y2), result in zip(coors, results):
            bbox = result['bboxes']
            cls = result['classes']
            conf = result['confs']

            bbox += (x1, y1, x1, y1)
            bboxes.append(bbox)
            classes.append(cls)
            confs.append(conf)

        bboxes = np.concatenate(bboxes)
        classes = np.concatenate(classes)
        confs = np.concatenate(confs)

        keep = cv_utils.non_max_suppression(bboxes, confs, object_detection.Iou().iou)
        bboxes = bboxes[keep]
        classes = classes[keep]
        confs = confs[keep]

        return dict(
            bboxes=bboxes,
            classes=classes,
            confs=confs
        )

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, **kwargs) -> List[dict]:
        images = objs[0][start_idx: end_idx]
        images = [os_lib.loader.load_img(image, channel_fixed_3=True) if isinstance(image, str) else image for image in images]
        rets = []
        for image in images:
            rets.append(dict(
                image=image
            ))
        return rets


class OdDataProcess(DataHooks):
    train_dataset_ins = OdDataset
    val_dataset_ins = OdDataset


class Voc(OdDataProcess):
    dataset_version = 'Voc2012'
    data_dir = 'data/VOC2012'
    train_data_num = None
    val_data_num = None

    input_size = 512
    in_ch = 3
    n_classes = 20

    cls_alias: dict

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.cv_data_parse.datasets.Voc import Loader

        loader = Loader(self.data_dir)
        self.cls_alias = loader.classes

        if train:
            return loader(
                set_type=DataRegister.TRAIN, image_type=DataRegister.PATH, generator=False,
                task='',
                max_size=self.train_data_num
            )[0]

        else:
            return loader(
                set_type=DataRegister.VAL, image_type=DataRegister.PATH, generator=False,
                task='',
                max_size=self.val_data_num,
            )[0]

    aug = RandomApply([geometry.HFlip()])

    post_aug = Apply([
        scale.LetterBox(),
        # pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret, train=True) -> dict:
        if train:
            ret.update(self.aug(**ret))

        ret.update(dst=self.input_size)
        ret.update(self.post_aug(**ret))
        return ret

    def val_data_restore(self, ret):
        ret = scale.LetterBox().restore(ret)
        return ret


class FastererRCNN_Voc(OdProcess, Voc):
    """
    Usage:
        .. code-block:: python

            from bundles.object_detect import FastererRCNN_Voc as Process

            Process(device=0).run(max_epoch=150, train_batch_size=32, accumulate=192)
            {'score': 0.3910}
    """
    model_version = 'FastererRCNN'
    input_size = 800

    def set_model(self):
        from models.object_detection.FasterRCNN import Model

        self.model = Model(
            in_module_config=dict(in_ch=self.in_ch, input_size=self.input_size),
            n_classes=self.n_classes
        )


class YoloV5(OdProcess):
    model_version = 'YoloV5'
    config_version = 'yolov5l'

    def set_model(self):
        """use auto anchors
        from models.object_detection.YoloV5 import Model, Config
        head_config = Config.auto_anchors(self.get_train_data(), self.input_size)
        self.model = Model(
            self.n_classes,
            in_module_config=dict(in_ch=self.in_ch, input_size=self.input_size),
            head_config=head_config
        )
        """
        from models.object_detection.YoloV5 import Model, Config

        model_config = Config.get(self.config_version)
        model_config.update(
            n_classes=self.n_classes
        )
        in_module_config = model_config['in_module_config']
        in_module_config.update(
            in_ch=self.in_ch,
            input_size=self.input_size
        )

        self.model = Model(**model_config)

    def set_optimizer(self, lr=0.01, momentum=0.937, **kwargs):
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g[0].append(v.weight)

        weight_decay = 0.0005

        self.optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        self.optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})  # add g0 with weight_decay
        self.optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)

        del g


class Yolov5Aug(OdDataProcess):
    """use Mosaic data augment"""
    aug = RandomApply([geometry.HFlip()])

    post_aug = Apply([
        scale.LetterBox(),
        # pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret, train=True) -> dict:
        if train:
            ret.update(self.aug(**ret))
        else:
            ret.update(dst=self.input_size)
            ret.update(self.post_aug(**ret))
        return ret

    mosaic_prob = 0.5

    def complex_data_augment(self, idx, data, base_process):
        """

        Args:
            idx:
            data:
            base_process:
                self.train_data_augment()

        """
        if np.random.random() > self.mosaic_prob:
            idxes = [idx] + list(np.random.choice(range(len(data)), 3, replace=False))
            rets = []
            for idx in idxes:
                ret = base_process(idx)
                rets.append(ret)

            image_list = [ret['image'] for ret in rets]
            bboxes_list = [ret['bboxes'] for ret in rets]
            classes_list = [ret['classes'] for ret in rets]
            img_size = np.max([img.shape[:2] for img in image_list])
            ret = complex.Mosaic4(img_size=img_size)(
                image_list=image_list,
                bboxes_list=bboxes_list,
                classes_list=classes_list
            )
        else:
            ret = base_process(idx)

        ret.update(dst=self.input_size)
        ret.update(self.post_aug(**ret))

        return ret

    def val_data_restore(self, ret):
        ret = scale.LetterBox().restore(ret)
        return ret


class YoloV5_Voc(YoloV5, Voc, Yolov5Aug):
    """
    Usage:
        .. code-block:: python

            from bundles.object_detect import YoloV5_Voc as Process

            Process(device=0).run(max_epoch=150, train_batch_size=32, accumulate=192)
            {'score': 0.3529}
    """


class Yolov5Dataset(Yolov5Aug):
    """supported official data type of yolov5
    see https://github.com/ultralytics/yolov5 to get more information"""

    dataset_version = 'yolov5'
    data_dir = 'yolov5/data_mapping'

    input_size = 640  # special input_size from official yolov5

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.cv_data_parse.datasets.YoloV5 import Loader, DataRegister

        def convert_func(ret):
            if isinstance(ret['image'], np.ndarray):
                h, w, c = ret['image'].shape
                ret['bboxes'] = cv_utils.CoordinateConvert.mid_xywh2top_xyxy(ret['bboxes'], wh=(w, h), blow_up=True)

            return ret

        loader = Loader(self.data_dir)
        loader.on_end_convert = convert_func

        if train:
            return loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False, **kwargs)[0]
        else:
            return loader(set_type=DataRegister.VAL, image_type=DataRegister.ARRAY, generator=False, **kwargs)[0]


class YoloV5_yolov5(YoloV5, Yolov5Dataset):
    """
    Usage:
        .. code-block:: python

        from bundles.object_detection import YoloV5_yolov5 as Process

        max_epoch = 21
        train_batch_size = 32
        predict_batch_size = None
        check_period = 3
        process = Process(
            # use_wandb=True,
            model_version='xxx',
            dataset_version='xxx',
            data_dir='xxx',
            classes=[...],
        )

        process.init()

        ####### train ###########
        process.fit(
            max_epoch=max_epoch, batch_size=train_batch_size, check_period=check_period,
            data_get_kwargs=dict(...),
            dataloader_kwargs=dict(num_workers=min(train_batch_size, 16)),
            metric_kwargs=dict(is_visualize=True, max_vis_num=8),
        )
        process.save(process.default_model_path)

        ######## val ##########
        process.load(f'{process.work_dir}/last.pth')

        r = process.metric(
            batch_size=predict_batch_size or train_batch_size,
            num_workers=16,
            is_visualize=True,
            max_vis_num=8,
            data_get_kwargs=dict(...),
        )
        for k, v in r.items():
            print(k)
            print(v)
    """

    def __init__(self, classes=None, **kwargs):
        kwargs.setdefault('n_classes', len(classes))
        super().__init__(**kwargs)


class PPOCRv4Det(Process):
    model_version = 'PPOCRv4_det'
    config_version = 'teacher'

    def set_model(self):
        from models.object_detection.PPOCRv4_det import Model, Config

        self.model = Model(**Config.get(self.config_version))

    def load_pretrained(self):
        from models.object_detection.PPOCRv4_det import WeightConverter

        state_dict = torch_utils.Load.from_file(self.pretrained_model)
        if self.config_version == 'teacher':
            state_dict = WeightConverter.from_teacher(state_dict)
        else:
            state_dict = WeightConverter.from_student(state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        # so silly that, import paddle will clear the logger settings, so reinit the logger
        log_utils.logger_init()

    def on_train_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs, train=True)

        output = self.model(**inputs)

        return output

    def metric(self, *args, **kwargs):
        process_results = self.predict(**kwargs)

        ap = object_detection.AP(iou_method=object_detection.PolygonIou().iou)
        metric_results = {}
        for name, results in process_results.items():
            gt_rets = results['trues']
            det_rets = results['preds']
            gt_segs = [ret['segmentations'] for ret in gt_rets]
            det_segs = [ret['segmentations'] for ret in det_rets]
            rets = ap.mAP_thres(gt_segs, det_segs)['']

            metric_results[name] = dict(
                result=rets,
                score=rets['ap']
            )

        return metric_results

    def get_model_inputs(self, loop_inputs, train=True):
        r = dict()
        if train:
            label_threshold_map = []
            label_threshold_mask = []
            label_shrink_map = []
            label_shrink_mask = []
            for ret in loop_inputs:
                image = ret['image']
                segmentations = ret['segmentations']

                tmp = Apply([
                    channel.CHW2HWC(),
                    pixel_perturbation.BorderMap()
                ])(image=image, segmentations=segmentations)
                label_threshold_map.append(tmp['mapping'])
                label_threshold_mask.append(tmp['mask'])

                tmp = Apply([
                    channel.CHW2HWC(),
                    pixel_perturbation.ShrinkMap()
                ])(image=image, segmentations=segmentations)
                label_shrink_map.append(tmp['mapping'])
                label_shrink_mask.append(tmp['mask'])

            label_threshold_map = torch.stack([torch.from_numpy(i) for i in label_threshold_map]).to(self.device, non_blocking=True, dtype=torch.float)
            label_threshold_mask = torch.stack([torch.from_numpy(i) for i in label_threshold_mask]).to(self.device, non_blocking=True, dtype=torch.float)
            label_shrink_map = torch.stack([torch.from_numpy(i) for i in label_shrink_map]).to(self.device, non_blocking=True, dtype=torch.float)
            label_shrink_mask = torch.stack([torch.from_numpy(i) for i in label_shrink_mask]).to(self.device, non_blocking=True, dtype=torch.float)

            r.update(
                label_list=(
                    label_threshold_map,
                    label_threshold_mask,
                    label_shrink_map,
                    label_shrink_mask,
                )
            )

        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
        images = torch.stack(images)
        r.update(x=images)

        return r

    def on_val_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs, train=False)

        model_results = {}
        for name, model in self.models.items():
            outputs = model(**inputs)
            outputs = [{k: v.to('cpu').numpy() if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in outputs]

            preds = []
            for (output, ret) in zip(outputs, loop_inputs):
                output = configs.ConfigObjParse.merge_dict(ret, output)
                output = self.val_data_restore(output)
                preds.append(dict(
                    segmentations=output['segmentations'],
                ))

            model_results[name] = dict(
                outputs=outputs,
                preds=preds,
            )

        return model_results

    def on_val_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        loop_inputs = loop_objs['loop_inputs']
        model_results = loop_objs['model_results']

        for name, results in model_results.items():
            r = process_results.setdefault(name, dict())
            trues = r.setdefault('trues', [])
            preds = r.setdefault('preds', [])
            for ret, pred in zip(loop_inputs, results['preds']):
                trues.append(dict(
                    _id=ret['_id'],
                    segmentations=ret['ori_segmentations'],
                ))
                pred['_id'] = ret['_id']
                preds.append(pred)

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, **kwargs) -> List[dict]:
        images = objs[0][start_idx: end_idx]
        ids = [Path(image).name if isinstance(image, str) else f'{i}.png' for i, image in zip(range(start_idx, end_idx), images)]
        images = [os_lib.loader.load_img(image, channel_fixed_3=True) if isinstance(image, str) else image for image in images]
        rets = []
        for _id, image in zip(ids, images):
            rets.append(dict(
                _id=_id,
                image=image
            ))
        return rets

    def on_predict_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        loop_inputs = loop_objs['loop_inputs']
        model_results = loop_objs['model_results']

        for name, results in model_results.items():
            r = process_results.setdefault(name, dict())
            preds = r.setdefault('preds', [])
            for ret, pred in zip(loop_inputs, results['preds']):
                pred['_id'] = ret['_id']
                preds.append(pred)


class IcdarDataset(BaseImgDataset):
    def process_one(self, idx):
        ret = copy.deepcopy(self.iter_data[idx])
        if isinstance(ret['image'], str):
            ret['image_path'] = ret['image']
            ret['image'] = cv2.imread(ret['image'])

        ret['ori_image'] = ret['image']
        ret['ori_segmentations'] = ret['segmentations']
        ret['idx'] = idx

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret


class Icdar(DataHooks):
    train_dataset_ins = IcdarDataset
    val_dataset_ins = IcdarDataset

    dataset_version = 'Icdar'
    data_dir = 'data/icdar2015'
    train_data_num = None
    val_data_num = None

    input_size = 960
    in_ch = 3

    def get_data(self, *args, train=True, **kwargs) -> Optional[Iterable | Dataset | List[Dataset]]:
        from data_parse.cv_data_parse.datasets.Icdar import Loader

        loader = Loader(self.data_dir)

        if train:
            return loader(
                set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False,
                task='',
                max_size=self.train_data_num
            )[0]

        else:
            return loader(
                set_type=DataRegister.VAL, image_type=DataRegister.ARRAY, generator=False,
                task='',
                max_size=self.val_data_num,
            )[0]

    aug = RandomApply([
        pixel_perturbation.GaussNoise()
    ])

    post_aug = Apply([
        scale.LetterBox(
            fill=(0, 0, 0),
            interpolation=1
        ),
        # channel.Keep3Dims(),
        pixel_perturbation.MinMax(),
        pixel_perturbation.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret, train=True) -> dict:
        if train:
            ret.update(self.aug(**ret))

        ret.update(dst=self.input_size)
        ret.update(self.post_aug(**ret))
        return ret

    def val_data_restore(self, ret):
        ret = scale.LetterBox().restore(ret)
        return ret


class PPOCRv4Det_Icdar(PPOCRv4Det, Icdar):
    """
    from bundles.object_detection import PPOCRv4Det_Icdar as Process

    model_dir = 'xxx'
    process = Process(
        config_version='student',
        pretrained_model=f'{model_dir}/ch_PP-OCRv4_det_train/best_accuracy.pdparams',
        # config_version='teacher',
        # pretrained_model=f'{model_dir}/ch_PP-OCRv4_det_server_train/best_accuracy.pdparams',
    )
    process.init()

    process.fit(
        use_ema=True,
        use_scheduler=True,
        scheduler_strategy='step',
        batch_size=16,
        metric_kwargs=dict(
            is_visualize=True,
            max_vis_num=10
        )
    )

    process.single_predict('xxx.png')
    """
