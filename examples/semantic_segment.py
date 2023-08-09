import cv2
import copy
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.os_lib import MemoryInfo
from utils.torch_utils import EarlyStopping
from utils.visualize import get_color_array
from metrics import classifier, mulit_classifier
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, channel, RandomApply, Apply
from data_parse import DataRegister
from data_parse.cv_data_parse.base import DataVisualizer
from .base import Process, BaseDataset
from torch.nn import functional as F
from PIL import Image
import math


class SegDataset(BaseDataset):
    def process_one(self, idx):
        ret = copy.deepcopy(self.data[idx])
        if isinstance(ret['image'], str):
            ret['image_path'] = ret['image']
            ret['image'] = cv2.imread(ret['image'])
            ret['pix_image_path'] = ret['pix_image']
            # note that, use PIL.Image to get the label image
            ret['pix_image'] = np.array(Image.open(ret['pix_image']))

        ret['ori_image'] = ret['image']
        ret['ori_pix_image'] = ret['pix_image']
        ret['idx'] = idx

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret


class SegProcess(Process):
    dataset = SegDataset

    def fit(self, dataset, max_epoch, batch_size, save_period=None, **dataloader_kwargs):
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
        lrf = 0.01

        # lf = lambda x: (1 - x / max_epoch) * (1.0 - lrf) + lrf
        # cos_lr
        lf = lambda x: ((1 - math.cos(x * math.pi / max_epoch)) / 2) * (lrf - 1) + 1

        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        scheduler.last_epoch = -1

        scaler = torch.cuda.amp.GradScaler(enabled=True)
        stopper = EarlyStopping(patience=10, min_epoch=10, stdout_method=self.logger.info)
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

                pbar.set_postfix({
                    'loss': f'{loss.item():.06}',
                    'mean_loss': f'{total_loss / total_batch:.06}',
                    # 'cpu_info': MemoryInfo.get_process_mem_info(),
                    # 'gpu_info': MemoryInfo.get_gpu_mem_info()
                })

            if save_period and i % save_period == save_period - 1:
                self.save(f'{self.model_dir}/{self.dataset_version}/last.pth')

                val_data = self.get_val_data()
                val_dataset = self.dataset(val_data, augment_func=self.val_data_augment)
                result = self.metric(val_dataset, batch_size)
                score = result['score']
                self.logger.info(f"epoch: {i}, score: {score}")

                if score > max_score:
                    self.save(f'{self.model_dir}/{self.dataset_version}/best.pth')
                    max_score = score

                if stopper(epoch=i, fitness=score):
                    break

    def predict(self, dataset, batch_size=16, visualize=False, max_vis_num=None, save_ret_func=None, **dataloader_kwargs):
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

                if visualize:
                    outputs = []
                    for ret, output in zip(rets, pred_images):
                        ret.pop('bboxes')

                        ori_pix_image = ret['ori_pix_image']
                        pred_image = np.zeros((*output.shape, 3), dtype=output.dtype) + 255
                        true_image = np.zeros((*ori_pix_image.shape, 3), dtype=output.dtype) + 255
                        for i in range(self.out_features + 1):
                            pred_image[output == i] = get_color_array(i)
                            true_image[ori_pix_image == i] = get_color_array(i)
                            # print(i, get_color_array(i))

                        det_ret = {'image': pred_image, **ret}
                        det_ret = self.val_data_restore(det_ret)
                        outputs.append(det_ret)

                        ret['image'] = ret['ori_image']
                        ret['pix_image'] = true_image

                    n = min(batch_size, max_vis_num - vis_num)
                    if n > 0:
                        DataVisualizer(self.save_result_dir, stdout_method=self.logger.info)(rets[:n], outputs[:n])
                        vis_num += n

        if save_ret_func:
            save_ret_func(pred)

        return true, pred

    def metric(self, dataset, batch_size=16, **kwargs):
        true, pred = self.predict(dataset, batch_size, **kwargs)
        # ignore background
        pred = np.concatenate(pred).flatten() - 1
        true = np.concatenate(true).flatten() - 1

        result = mulit_classifier.TopMetric(n_class=self.out_features, ignore_class=(-1, 254)).f1(true, pred)

        result.update(
            score=result['f']
        )

        return result


class Voc(Process):
    def data_augment(self, ret):
        aug = Apply([
            # scale.LetterBox(),
            scale.Jitter(),
            # pixel_perturbation.MinMax(),
            channel.HWC2CHW()
        ])

        random_aug = RandomApply([
            geometry.HFlip(),
            geometry.VFlip(),
        ])

        ret.update(dst=self.input_size)
        ret.update(aug(**ret))
        ret.update(random_aug(**ret))

        pix_image = ret['pix_image']
        # note that, use cv2.INTER_NEAREST mode to resize
        # pix_image = scale.LetterBox(interpolation=1, fill=255).apply_image(pix_image, ret)
        pix_image = scale.Jitter(interpolation=1, fill=255).apply_image(pix_image, ret)
        pix_image = random_aug.apply_image(pix_image, ret)
        ret['pix_image'] = pix_image
        return ret

    def get_train_data(self):
        from data_parse.cv_data_parse.Voc import Loader, DataRegister, SEG_CLS

        loader = Loader('data/VOC2012')
        return loader(set_type=DataRegister.TRAIN, generator=False, image_type=DataRegister.PATH, set_task=SEG_CLS)[0]

    def val_data_augment(self, ret):
        aug = Apply([
            scale.LetterBox(),
            # scale.Jitter(self.input_size),
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

        loader = Loader('data/VOC2012')
        return loader(set_type=DataRegister.VAL, generator=False, image_type=DataRegister.PATH, set_task=SEG_CLS)[0]
        # return loader(set_type=DataRegister.TRAIN, generator=False, image_type=DataRegister.PATH, set_task=SEG_CLS)[0]


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
