import math
import cv2
import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from utils.visualize import get_color_array
from .base import Process, BaseDataset
from metrics import classifier, mulit_classifier
from data_parse.cv_data_parse.base import DataVisualizer
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, channel, pixel_perturbation, RandomApply, Apply


class SegDataset(BaseDataset):
    def process_one(self, idx):
        ret = copy.deepcopy(self.data[idx])
        if isinstance(ret['image'], str):
            ret['image_path'] = ret['image']
            ret['image'] = cv2.imread(ret['image'])
            ret['pix_image_path'] = ret['pix_image']
            # note that, use PIL.Image to get the label image
            ret['pix_image'] = np.asarray(Image.open(ret['pix_image']))

        ret['ori_image'] = ret['image']
        ret['ori_pix_image'] = ret['pix_image']
        ret['idx'] = idx

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret


class SegProcess(Process):
    def fit(self, max_epoch, batch_size, save_period=None, metric_kwargs=dict(), **dataloader_kwargs):
        train_dataloader, val_dataloader, metric_kwargs = self.on_train_start(batch_size, metric_kwargs, **dataloader_kwargs)

        lrf = 0.01

        # lf = lambda x: (1 - x / max_epoch) * (1.0 - lrf) + lrf
        lf = lambda x: ((1 - math.cos(x * math.pi / max_epoch)) / 2) * (lrf - 1) + 1  # cos_lr

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        scheduler.last_epoch = -1

        scaler = torch.cuda.amp.GradScaler(enabled=True)
        accumulate = 64 // batch_size
        j = 0

        for i in range(self.start_epoch, max_epoch):
            self.model.train()
            pbar = tqdm(train_dataloader, desc=f'train {i}/{max_epoch}')
            total_loss = 0
            total_batch = 0
            losses = None

            for rets in pbar:
                images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images = torch.stack(images)

                pix_images = [torch.from_numpy(ret.pop('pix_image')).to(self.device, non_blocking=True, dtype=torch.long) for ret in rets]
                pix_images = torch.stack(pix_images)

                images = images / 255

                with torch.cuda.amp.autocast(True):
                    output = self.model(images, pix_images)
                loss = output['loss']

                scaler.scale(loss).backward()
                if j % accumulate == 0:
                    scaler.unscale_(self.optimizer)  # unscale gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
                    scaler.step(self.optimizer)  # optimizer.step
                    scaler.update()
                    self.optimizer.zero_grad()

                j += 1
                total_loss += loss.item()
                total_batch += len(rets)
                mean_loss = total_loss / total_batch

                losses = {
                    'loss': loss.item(),
                    'mean_loss': mean_loss,
                }
                # mem_info = {
                #     'cpu_info': log_utils.MemoryInfo.get_process_mem_info(),
                #     'gpu_info': log_utils.MemoryInfo.get_gpu_mem_info()
                # }

                pbar.set_postfix({
                    **losses,
                    # **mem_info
                })

            scheduler.step()
            if self.on_train_epoch_end(i, save_period, val_dataloader,
                                       losses=losses,
                                       **metric_kwargs):
                break

    def metric(self, *args, **kwargs):
        true, pred = self.predict(*args, **kwargs)
        # ignore background
        pred = np.concatenate(pred).flatten() - 1
        true = np.concatenate(true).flatten() - 1

        result = mulit_classifier.TopMetric(n_class=self.out_features, ignore_class=(-1, 254)).f1(true, pred)

        result.update(
            score=result['f']
        )

        return result

    def predict(self, dataset, batch_size=16, cur_epoch=-1, visualize=False, max_vis_num=None, save_ret_func=None, **dataloader_kwargs):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            **dataloader_kwargs
        )

        self.model.to(self.device)
        pred = []
        true = []
        max_vis_num = max_vis_num or float('inf')
        vis_num = 0

        with torch.no_grad():
            self.model.eval()
            for rets in tqdm(dataloader):
                images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images = torch.stack(images)
                images = images / 255

                pred_images = self.model(images).cpu().detach().numpy().astype(np.uint8)
                pred.append(pred_images)
                true.append([ret.pop('pix_image') for ret in rets])

                vis_num = self.on_val_step_end(rets, pred_images, cur_epoch, visualize, batch_size, max_vis_num, vis_num)

        if save_ret_func:
            save_ret_func(pred)

        return true, pred

    def on_val_step_end(self, rets, pred_images, cur_epoch, visualize, batch_size, max_vis_num, vis_num):
        if visualize:
            n = min(batch_size, max_vis_num - vis_num)
            if n > 0:
                outputs = []
                for ret, output in zip(rets, pred_images):
                    ret.pop('bboxes')

                    ori_pix_image = ret['ori_pix_image']
                    pred_image = np.zeros((*output.shape, 3), dtype=output.dtype) + 255
                    true_image = np.zeros((*ori_pix_image.shape, 3), dtype=output.dtype) + 255
                    for i in range(self.out_features + 1):
                        pred_image[output == i] = get_color_array(i)
                        true_image[ori_pix_image == i] = get_color_array(i)

                    det_ret = {'image': pred_image, **ret}
                    det_ret = self.val_data_restore(det_ret)
                    outputs.append(det_ret)

                    ret['image'] = ret['ori_image']
                    ret['pix_image'] = true_image

                cache_image = DataVisualizer(f'{self.save_result_dir}/{cur_epoch}', verbose=False, pbar=False)(rets[:n], outputs[:n], return_image=True)
                self.log_info.setdefault('val_image', []).extend([self.wandb.Image(img, mode='BGR', caption=Path(r['_id']).stem) for img, r in zip(cache_image, rets)])
                vis_num += n

        return vis_num


class Voc(Process):
    data_dir = 'data/VOC2012'

    def data_augment(self, ret):
        random_aug = RandomApply([
            geometry.HFlip(),
            geometry.VFlip(),
        ])

        aug = Apply([
            # scale.LetterBox(),
            scale.Jitter(),
            # pixel_perturbation.MinMax(),
            channel.HWC2CHW()
        ])

        ret.update(random_aug(**ret))
        ret.update(RandomApply([pixel_perturbation.Jitter()])(**ret))
        ret.update(dst=self.input_size)
        ret.update(aug(**ret))

        pix_image = ret['pix_image']
        # note that, use cv2.INTER_NEAREST mode to resize
        # pix_image = scale.LetterBox(interpolation=1, fill=255).apply_image(pix_image, ret)
        pix_image = random_aug.apply_image(pix_image, ret)
        pix_image = scale.Jitter(interpolation=1, fill=255).apply_image(pix_image, ret)
        ret['pix_image'] = pix_image
        return ret

    def get_train_data(self):
        from data_parse.cv_data_parse.Voc import Loader, DataRegister, SEG_CLS

        loader = Loader(self.data_dir)
        return loader(set_type=DataRegister.TRAIN, generator=False, image_type=DataRegister.PATH, set_task=SEG_CLS)[0]

    def val_data_augment(self, ret):
        aug = Apply([
            scale.LetterBox(),
            # pixel_perturbation.MinMax(),
            channel.HWC2CHW()
        ])
        ret.update(dst=self.input_size)
        ret.update(aug(**ret))
        pix_image = ret['pix_image']
        pix_image = scale.LetterBox(interpolation=1, fill=255).apply_image(pix_image, ret)
        ret['pix_image'] = pix_image
        return ret

    def val_data_restore(self, ret):
        ret = scale.LetterBox().restore(ret)
        return ret

    def get_val_data(self):
        from data_parse.cv_data_parse.Voc import Loader, DataRegister, SEG_CLS

        loader = Loader(self.data_dir)
        return loader(set_type=DataRegister.VAL, generator=False, image_type=DataRegister.PATH, set_task=SEG_CLS)[0]


class FCN_Voc(SegProcess, Voc):
    """
    Usage:
        .. code-block:: python

            from examples.semantic_segment import FCN_Voc as Process

            Process().run(max_epoch=1000)
            {'score': 0.1429}
    """

    def __init__(self, model_version='FCN', dataset_version='Voc',
                 input_size=512, in_ch=3, out_features=20, **kwargs):
        from models.semantic_segmentation.FCN import Model
        model = Model(
            in_ch=in_ch,
            input_size=input_size,
            out_features=out_features
        )
        super().__init__(
            model=model,
            optimizer=optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            out_features=out_features,
            **kwargs
        )


class Unet_Voc(SegProcess, Voc):
    """
    Usage:
        .. code-block:: python

            from examples.semantic_segment import Unet_Voc as Process

            Process().run(max_epoch=1000)
            {'score': 0.1973}
    """

    def __init__(self, model_version='Unet', dataset_version='Voc',
                 input_size=512, in_ch=3, out_features=20, **kwargs):
        from models.semantic_segmentation.Unet import Model
        model = Model(
            in_ch=in_ch,
            input_size=input_size,
            out_features=out_features
        )
        super().__init__(
            model=model,
            optimizer=optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            out_features=out_features,
            **kwargs
        )


class DeeplabV3_Voc(SegProcess, Voc):
    """
    Usage:
        .. code-block:: python

            from examples.semantic_segment import DeeplabV3_Voc as Process

            Process().run(max_epoch=1000)
            {'score': 0.3021}
    """

    def __init__(self, model_version='DeeplabV3', dataset_version='Voc',
                 input_size=512, in_ch=3, out_features=20, **kwargs):
        from models.semantic_segmentation.DeeplabV3 import Model
        model = Model(
            in_ch=in_ch,
            input_size=input_size,
            out_features=out_features
        )
        super().__init__(
            model=model,
            optimizer=optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            out_features=out_features,
            **kwargs
        )
