import copy
import cv2
import math
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply, Apply, channel, complex
from data_parse.cv_data_parse.base import DataRegister, DataVisualizer
from .base import Process, BaseDataset
from utils.torch_utils import EarlyStopping
from utils.visualize import ImageVisualize
from utils import configs, os_lib, converter
from metrics import object_detection


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
    dataset = OdDataset

    def get_train_data(self):
        """example"""
        from data_parse.cv_data_parse.Voc import Loader

        loader = Loader(f'data/VOC2012')
        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.PATH, generator=False, task='')[0]

        return data

    def data_augment(self, ret):
        ret.update(RandomApply([geometry.HFlip()])(**ret))
        ret.update(dst=self.input_size)
        ret.update(Apply([
            scale.LetterBox(),
            # pixel_perturbation.MinMax(),
            # pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            channel.HWC2CHW()
        ])(**ret))
        return ret

    def complex_data_augment(self, idx, data, base_process):
        return base_process(idx)

    def val_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(Apply([
            scale.LetterBox(),
            # pixel_perturbation.MinMax(),
            # pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            channel.HWC2CHW()
        ])(**ret))
        return ret

    def val_data_restore(self, ret):
        ret = scale.LetterBox().restore(ret)
        return ret

    def get_val_data(self):
        """example"""
        from data_parse.cv_data_parse.Voc import Loader

        loader = Loader(f'data/VOC2012')
        data = loader(set_type=DataRegister.VAL, image_type=DataRegister.PATH, generator=False, task='')[0]
        # data = data[:20]

        return data

    def fit(self, dataset, max_epoch, batch_size, save_period=None, **dataloader_kwargs):
        # sampler = distributed.DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset,
            shuffle=True,
            # sampler=sampler,
            pin_memory=True,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            **dataloader_kwargs
        )

        self.model.to(self.device)

        # optimizer = optim.Adam(self.model.parameters())
        # optimizer = optim.SGD(self.model.parameters(), lr=0.01)

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
        optimizer = optim.SGD(g[2], lr=0.01, momentum=0.937, nesterov=True)
        optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1]})  # add g1 (BatchNorm2d weights)

        del g

        lrf = 0.01

        # lf = lambda x: (1 - x / max_epoch) * (1.0 - lrf) + lrf

        # cos_lr
        lf = lambda x: ((1 - math.cos(x * math.pi / max_epoch)) / 2) * (lrf - 1) + 1

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        scheduler.last_epoch = -1

        scaler = torch.cuda.amp.GradScaler(enabled=True)
        stopper = EarlyStopping(patience=10, stdout_method=self.logger.info)

        max_score = -1
        accumulate = 64 / batch_size
        j = 0

        for i in range(max_epoch):
            self.model.train()
            pbar = tqdm(dataloader, desc=f'train {i}/{max_epoch}')
            total_loss = 0
            total_batch = 0

            for rets in pbar:
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
                loss = output['loss']

                scaler.scale(loss).backward()
                if j % accumulate == 0:
                    scaler.unscale_(optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
                    scaler.step(optimizer)  # optimizer.step
                    scaler.update()
                    optimizer.zero_grad()

                j += 1

                total_loss += loss.item()
                total_batch += len(rets)

                pbar.set_postfix({
                    'loss': f'{loss.item():.06}',
                    'mean_loss': f'{total_loss / total_batch:.06}',
                    # 'cpu_info': MemoryInfo.get_process_mem_info(),
                    # 'gpu_info': MemoryInfo.get_gpu_mem_info()
                })

            if save_period and i % save_period == save_period - 1:
                self.save(f'{self.model_dir}/{self.dataset_version}_last.pth')

                val_data = self.get_val_data()
                val_dataset = self.dataset(val_data, augment_func=self.val_data_augment)
                result = self.metric(val_dataset, batch_size, **dataloader_kwargs)
                score = result['score']
                self.logger.info(f"epoch: {i}, score: {score}")

                if score > max_score:
                    self.save(f'{self.model_dir}/{self.dataset_version}_best.pth')
                    max_score = score

                if stopper(epoch=i, fitness=score):
                    break

        scheduler.step()

    def predict(self, dataset, batch_size=128, visualize=False, save_ret_func=None, **dataloader_kwargs):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            **dataloader_kwargs
        )

        self.model.to(self.device)
        gt_rets, det_rets = [], []

        with torch.no_grad():
            self.model.eval()
            for rets in tqdm(dataloader, desc='val'):
                images = [torch.Tensor(ret['image']).to(self.device) for ret in rets]
                images = torch.stack(images)
                images = images / 255

                outputs = self.model(images)
                outputs = [{k: v.to('cpu').numpy() for k, v in t.items()} for t in outputs]

                for i in range(len(images)):
                    output = outputs[i]
                    ret = rets[i]

                    output = configs.merge_dict(ret, output)
                    ret = self.val_data_restore(ret)
                    output = self.val_data_restore(output)

                    gt_rets.append(dict(
                        _id=ret['_id'],
                        bboxes=ret['bboxes'],
                        classes=ret['classes'],
                    ))

                    det_rets.append(dict(
                        _id=ret['_id'],
                        bboxes=output['bboxes'],
                        classes=output['classes'],
                        confs=output['confs']
                    ))

                if visualize:
                    for ret, output in zip(rets, output):
                        ret['image'] = ret['ori_image']
                        output['image'] = ret['ori_image']
                    DataVisualizer(self.save_result_dir)(rets, outputs)

        if save_ret_func:
            save_ret_func(det_rets)

        return gt_rets, det_rets

    def metric(self, dataset, batch_size=128, **kwargs):
        gt_rets, det_rets = self.predict(dataset, batch_size, **kwargs)
        df = object_detection.quick_metric(gt_rets, det_rets, save_path=f'{self.model_dir}/{self.dataset_version}.csv', verbose=False)

        result = dict(
            per_class_result=df,
            score=df['ap']['mean']
        )

        return result


class FastererRCNN_Voc(OdProcess):
    """
    Usage:
        .. code-block:: python

            from examples.object_detect import FastererRCNN_Voc as Process

            Process().run()
            {'score': 0.3910}
    """

    def __init__(self, device=1):
        from models.object_detection.FasterRCNN import Model

        in_ch = 3
        input_size = 800
        n_classes = 20

        super().__init__(
            model=Model(
                in_module_config=dict(in_ch=in_ch, input_size=input_size),
                n_classes=n_classes
            ),
            model_version='FastererRCNN',
            dataset_version='Voc2012',
            input_size=input_size,
            device=device
        )


class YoloV5(OdProcess):
    def data_augment(self, ret):
        ret.update(RandomApply([geometry.HFlip()])(**ret))
        return ret

    def complex_data_augment(self, idx, data, base_process):
        if np.random.random() > 0.5:
            idxes = [idx] + list(np.random.choice(range(len(data)), 3, replace=False))
            rets = []
            for idx in idxes:
                ret = base_process(idx)
                rets.append(ret)

            image_list = [ret['image'] for ret in rets]
            bboxes_list = [ret['bboxes'] for ret in rets]
            classes_list = [ret['classes'] for ret in rets]
            img_size = np.max([img.shape[:2] for img in image_list])
            ret = complex.Mosaic(img_size=img_size)(
                image_list=image_list,
                bboxes_list=bboxes_list,
                classes_list=classes_list
            )
        else:
            ret = base_process(idx)

        ret.update(dst=self.input_size)
        ret.update(Apply([
            scale.LetterBox(),
            # pixel_perturbation.MinMax(),
            # pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            channel.HWC2CHW()
        ])(**ret))

        return ret

    def val_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(Apply([
            scale.LetterBox(),
            # pixel_perturbation.MinMax(),
            # pixel_perturbation.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            channel.HWC2CHW()
        ])(**ret))
        return ret


class YoloV5_Voc(YoloV5):
    """
    Usage:
        .. code-block:: python

            from examples.object_detect import YoloV5_Voc as Process

            Process().run(max_epoch=500)
            {'score': 0.3529}
    """

    def __init__(self, device=1):
        from models.object_detection.YoloV5 import Model

        in_ch = 3
        input_size = 640
        n_classes = 20

        super().__init__(
            model=Model(
                n_classes,
                in_module_config=dict(in_ch=in_ch, input_size=input_size),
            ),
            model_version='YoloV5',
            dataset_version='Voc2012',
            input_size=input_size,
            device=device
        )


class YoloV5_yolov5(YoloV5):
    def __init__(self, classes, device=1):
        from models.object_detection.YoloV5 import Model

        in_ch = 3
        input_size = 640
        n_classes = len(classes)

        super().__init__(
            model=Model(
                n_classes,
                in_module_config=dict(in_ch=in_ch, input_size=input_size),
            ),
            model_version='YoloV5',
            dataset_version='yolov5',
            input_size=input_size,
            device=device
        )

    def get_train_data(self):
        from data_parse.cv_data_parse.YoloV5 import Loader, DataRegister

        loader = Loader('yolov5/data_mapping')
        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.PATH, generator=False, sub_dir='')[0]

        return data

    def data_augment(self, ret):
        image = ret['image']
        ret['bboxes'] = converter.CoordinateConvert.mid_xywh2top_xyxy(ret['bboxes'], wh=(image.shape[1], image.shape[0]), blow_up=True)
        return super().data_augment(ret)

    def get_val_data(self):
        from data_parse.cv_data_parse.YoloV5 import Loader, DataRegister

        loader = Loader('yolov5/data_mapping')
        data = loader(set_type=DataRegister.VAL, image_type=DataRegister.PATH, generator=False, sub_dir='')[0]
        # data = data[:20]

        return data

    def val_data_augment(self, ret):
        image = ret['image']
        ret['bboxes'] = converter.CoordinateConvert.mid_xywh2top_xyxy(ret['bboxes'], wh=(image.shape[1], image.shape[0]), blow_up=True)
        return super().val_data_augment(ret)
