import cv2
import copy
import math
import numpy as np
import torch
from torch import optim, nn
from metrics import object_detection
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, channel, RandomApply, Apply, complex
from data_parse import DataRegister
from pathlib import Path
from data_parse.cv_data_parse.base import DataVisualizer
from processor import Process, DataHooks, bundled, BaseDataset
from utils import configs, cv_utils


class OdDataset(BaseDataset):
    def process_one(self, idx):
        ret = copy.deepcopy(self.data[idx])
        if isinstance(ret['image'], str):
            ret['image_path'] = ret['image']
            ret['image'] = cv2.imread(ret['image'])

        ret['ori_image'] = ret['image']
        ret['ori_bboxes'] = ret['bboxes']
        ret['idx'] = idx

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret


class OdProcess(Process):

    def on_train_start(self, container, max_epoch=None, **kwargs):
        super().on_train_start(container, **kwargs)
        self.set_scheduler(max_epoch=max_epoch)
        self.set_scaler()

    def on_train_step(self, rets, container, **kwargs) -> dict:
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        gt_boxes = [torch.from_numpy(ret['bboxes']).to(self.device) for ret in rets]
        gt_cls = [torch.from_numpy(ret['classes']).to(self.device) for ret in rets]
        images = torch.stack(images)

        # note that, if the images have the same shape, minmax after stack if possible
        # it can reduce about 20 seconds per epoch to voc dataset
        images = images / 255

        # note that, amp method can make the model run in dtype of half
        # even though input has dtype of torch.half and weight has dtype of torch.float
        # so that, it would run in lower memory and cost less time
        with torch.cuda.amp.autocast(True):
            output = self.model(images, gt_boxes, gt_cls)

        return output

    def on_backward(self, output, container, batch_size=16, accumulate=64, **kwargs):
        loss = output['loss']

        self.scaler.scale(loss).backward()
        if self.counters['total_nums'] % accumulate < batch_size:
            self.scaler.unscale_(self.optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()

    def on_train_epoch_end(self, *args, **kwargs) -> bool:
        self.scheduler.step()
        return super().on_train_epoch_end(*args, **kwargs)

    def metric(self, **kwargs):
        container = self.predict(**kwargs)
        gt_rets = container['trues']
        det_rets = container['preds']
        df = object_detection.EasyMetric(verbose=False).quick_metric(gt_rets, det_rets, save_path=f'{self.work_dir}/result.csv')

        log_info = {
            # 'table': self.wandb.Table(dataframe=df)
        }
        for i in df.index:
            name = i
            if hasattr(self, 'cls_alias'):
                try:
                    name = self.cls_alias[i]
                except:
                    pass
            log_info[f'metrics/{name}'] = df['ap'][i]

        self.trace(log_info, bundled.WANDB)

        result = dict(
            per_class_result=df,
            score=df['ap']['mean']
        )

        return result

    def on_val_step(self, rets, container, **kwargs) -> tuple:
        images = [torch.from_numpy(ret.pop('image')).to(self.device) for ret in rets]
        images = torch.stack(images)
        images = images / 255

        outputs = container['model'](images)
        outputs = [{k: v.to('cpu').numpy() for k, v in t.items()} for t in outputs]

        for i in range(len(images)):
            output = outputs[i]
            ret = rets[i]

            output = configs.merge_dict(ret, output)
            ret = self.val_data_restore(ret)
            output = self.val_data_restore(output)

            outputs[i] = output
            rets[i] = ret

            container['trues'].append(dict(
                _id=ret['_id'],
                bboxes=ret['bboxes'],
                classes=ret['classes'],
            ))

            container['preds'].append(dict(
                _id=ret['_id'],
                bboxes=output['bboxes'],
                classes=output['classes'],
                confs=output['confs']
            ))

        return rets, outputs

    def on_val_step_end(self, rets, outputs, container, is_visualize=False, batch_size=16, max_vis_num=None, **kwargs):
        if is_visualize:
            max_vis_num = max_vis_num or float('inf')
            n = min(batch_size, max_vis_num - self.counters['vis_num'])
            if n > 0:
                for ret, output in zip(rets, outputs):
                    ret['image'] = ret['ori_image']
                    output['image'] = ret['ori_image']

                cls_alias = self.__dict__.get('cls_alias')
                cache_image = DataVisualizer(f'{self.cache_dir}/{self.counters["epoch"]}', verbose=False, pbar=False)(
                    rets[:n], outputs[:n], return_image=True, cls_alias=cls_alias
                )
                self.get_log_trace(bundled.WANDB).setdefault('val_image', []).extend(
                    [self.wandb.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=Path(r['_id']).stem) for img, r in zip(cache_image, rets)]
                )
                self.counters['vis_num'] += n

    def fragment_predict(self, image: np.ndarray, **kwargs):
        images, coors = cv_utils.fragment_image(image, max_size=self.input_size, over_ratio=0.5, overlap_ratio=0.2)
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

        keep = cv_utils.non_max_suppression(bboxes, confs, object_detection.Iou.iou)
        bboxes = bboxes[keep]
        classes = classes[keep]
        confs = confs[keep]

        return dict(
            bboxes=bboxes,
            classes=classes,
            confs=confs
        )


class Voc(DataHooks):
    dataset_version = 'Voc2012'
    data_dir = 'data/VOC2012'
    train_data_num = None
    val_data_num = None

    input_size = 512
    in_ch = 3
    n_classes = 20

    cls_alias: dict

    def get_train_data(self):
        from data_parse.cv_data_parse.Voc import Loader

        loader = Loader(self.data_dir)
        self.cls_alias = loader.classes

        return loader(set_type=DataRegister.TRAIN, image_type=DataRegister.PATH, generator=False,
                      task='',
                      max_size=self.train_data_num
                      )[0]

    def get_val_data(self):
        from data_parse.cv_data_parse.Voc import Loader

        loader = Loader(self.data_dir)
        self.cls_alias = loader.classes

        return loader(set_type=DataRegister.VAL, image_type=DataRegister.PATH, generator=False,
                      task='',
                      max_size=self.val_data_num,
                      )[0]

    aug = Apply([
        scale.LetterBox(),
        # pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        channel.HWC2CHW()
    ])

    def train_data_augment(self, ret):
        ret.update(RandomApply([geometry.HFlip()])(**ret))
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))
        return ret

    def val_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))
        return ret

    def val_data_restore(self, ret):
        ret = scale.LetterBox().restore(ret)
        return ret


class FastererRCNN_Voc(OdProcess, Voc):
    """
    Usage:
        .. code-block:: python

            from examples.object_detect import FastererRCNN_Voc as Process

            Process().run()
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
    input_size = 640

    def set_model(self):
        # auto anchors
        # from models.object_detection.YoloV5 import auto_anchors
        # head_config = auto_anchors(self.get_train_data(), input_size)

        from models.object_detection.YoloV5 import Model
        self.model = Model(
            self.n_classes,
            in_module_config=dict(in_ch=self.in_ch, input_size=self.input_size),
            # head_config=head_config
        )

    def set_optimizer(self):
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

        self.optimizer = optim.SGD(g[2], lr=0.01, momentum=0.937, nesterov=True)
        self.optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})  # add g0 with weight_decay
        self.optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)

        del g


class VocWithYolov5Aug(Voc):
    def train_data_augment(self, ret):
        ret.update(RandomApply([geometry.HFlip()])(**ret))
        return ret

    mosaic_prob = 0.5

    def complex_data_augment(self, idx, data, base_process):
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
        ret.update(Apply([
            scale.LetterBox(),
            # pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            channel.HWC2CHW()
        ])(**ret))

        return ret


class YoloV5_Voc(YoloV5, VocWithYolov5Aug):
    """
    Usage:
        .. code-block:: python

            from examples.object_detect import YoloV5_Voc as Process

            Process().run(max_epoch=500)
            {'score': 0.3529}
    """


class Yolov5Dataset(DataHooks):
    """supported official data type of yolov5
    see https://github.com/ultralytics/yolov5 to get more information"""

    dataset_version = 'yolov5'
    data_dir = 'yolov5/data_mapping'

    def get_train_data(self):
        from data_parse.cv_data_parse.YoloV5 import Loader, DataRegister

        convert_func = lambda ret: cv_utils.CoordinateConvert.mid_xywh2top_xyxy(
            ret['bboxes'],
            wh=(ret['image'].shape[1], ret['image'].shape[0]),
            blow_up=True
        )

        loader = Loader(self.data_dir)
        loader.on_end_convert = convert_func
        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.PATH, generator=False, sub_dir='')[0]

        return data

    def get_val_data(self):
        from data_parse.cv_data_parse.YoloV5 import Loader, DataRegister

        convert_func = lambda ret: cv_utils.CoordinateConvert.mid_xywh2top_xyxy(
            ret['bboxes'],
            wh=(ret['image'].shape[1], ret['image'].shape[0]),
            blow_up=True
        )

        loader = Loader(self.data_dir)
        loader.on_end_convert = convert_func
        data = loader(set_type=DataRegister.VAL, image_type=DataRegister.PATH, generator=False, sub_dir='')[0]

        return data


class YoloV5_yolov5(YoloV5, Yolov5Dataset):
    def __init__(self, classes=None, **kwargs):
        kwargs.setdefault('n_classes', len(classes))
        super().__init__(**kwargs)