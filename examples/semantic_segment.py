import cv2
import copy
import numpy as np
import torch
from torch import optim, nn
from metrics import mulit_classifier
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, channel, RandomApply, Apply, complex, pixel_perturbation
from data_parse import DataRegister
from pathlib import Path
from PIL import Image
from data_parse.cv_data_parse.base import DataVisualizer
from processor import Process, DataHooks, bundled, BaseDataset
from utils.visualize import get_color_array


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
    def set_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    def on_train_start(self, container, batch_size=16, max_epoch=None, **kwargs):
        super().on_train_start(container, batch_size=batch_size, **kwargs)

        self.set_scheduler(max_epoch=max_epoch)
        self.set_scaler()

    def on_train_step(self, rets, container, **kwargs) -> dict:
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        images = torch.stack(images)

        pix_images = [torch.from_numpy(ret.pop('pix_image')).to(self.device, non_blocking=True, dtype=torch.long) for ret in rets]
        pix_images = torch.stack(pix_images)

        images = images / 255

        with torch.cuda.amp.autocast(True):
            output = self.model(images, pix_images)

        return output

    def on_backward(self, output, container, batch_size=16, accumulate=64, **kwargs):
        loss = output['loss']
        counters = container['counters']

        self.scaler.scale(loss).backward()
        if counters['total_nums'] % accumulate < batch_size:
            self.scaler.unscale_(self.optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()

    def on_train_epoch_end(self, *args, **kwargs) -> bool:
        self.scheduler.step()
        return super().on_train_epoch_end(*args, **kwargs)

    def metric(self, *args, **kwargs):
        container = self.predict(*args, **kwargs)
        # ignore background
        pred = np.concatenate(container['preds']).flatten() - 1
        true = np.concatenate(container['trues']).flatten() - 1

        result = mulit_classifier.TopMetric(n_class=self.out_features, ignore_class=(-1, 254)).f1(true, pred)

        result.update(
            score=result['f']
        )

        return result

    def on_val_step(self, rets, container, **kwargs) -> tuple:
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        images = torch.stack(images)
        images = images / 255

        outputs = self.model(images).cpu().detach().numpy().astype(np.uint8)
        container['preds'].append(outputs)
        container['trues'].append([ret.pop('pix_image') for ret in rets])

        return rets, outputs

    def on_val_step_end(self, rets, outputs, container, is_visualize=False, batch_size=16, max_vis_num=None, **kwargs):
        if is_visualize:
            max_vis_num = max_vis_num or float('inf')
            counters = container['counters']
            n = min(batch_size, max_vis_num - counters['vis_num'])
            if n > 0:
                det_rets = []
                for ret, output in zip(rets, outputs):
                    ret.pop('bboxes')

                    ori_pix_image = ret['ori_pix_image']
                    pred_image = np.zeros((*output.shape, 3), dtype=output.dtype) + 255
                    true_image = np.zeros((*ori_pix_image.shape, 3), dtype=output.dtype) + 255
                    for i in range(self.out_features + 1):
                        pred_image[output == i] = get_color_array(i)
                        true_image[ori_pix_image == i] = get_color_array(i)

                    det_ret = {'image': pred_image, **ret}
                    det_ret = self.val_data_restore(det_ret)
                    det_rets.append(det_ret)

                    ret['image'] = ret['ori_image']
                    ret['pix_image'] = true_image

                cache_image = DataVisualizer(f'{self.cache_dir}/{counters["epoch"]}', verbose=False, pbar=False)(rets[:n], det_rets[:n], return_image=True)
                self.get_log_trace(bundled.WANDB).setdefault('val_image', []).extend(
                    [self.wandb.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=Path(r['_id']).stem) for img, r in zip(cache_image, rets)]
                )
                counters['vis_num'] += n


class Voc(DataHooks):
    dataset_version = 'VOC2012'
    data_dir = 'data/VOC2012'
    input_size = 512
    in_ch = 3
    out_features = 20

    aug = Apply([
        scale.Proportion(choice_type=3),
        crop.Random(is_pad=False),
        # scale.LetterBox(),    # there are gray lines
    ])

    # note that, use cv2.INTER_NEAREST mode to resize
    pix_aug = Apply([
        scale.Proportion(choice_type=3, interpolation=1),
        crop.Random(is_pad=False, fill=255),
        # scale.LetterBox(interpolation=1, fill=255),    # there are gray lines
    ])

    rand_aug1 = RandomApply([
        geometry.HFlip(),
        geometry.VFlip(),
    ])
    rand_aug2 = RandomApply([pixel_perturbation.Jitter()])

    post_aug = Apply([
        # pixel_perturbation.MinMax(),
        # pixel_perturbation.Normalize(127.5, 127.5),
        channel.HWC2CHW()
    ])

    def get_train_data(self):
        from data_parse.cv_data_parse.Voc import Loader, DataRegister, SEG_CLS

        loader = Loader(self.data_dir)
        return loader(set_type=DataRegister.TRAIN, generator=False, image_type=DataRegister.PATH, set_task=SEG_CLS)[0]

    def get_val_data(self):
        from data_parse.cv_data_parse.Voc import Loader, DataRegister, SEG_CLS

        loader = Loader(self.data_dir)
        return loader(set_type=DataRegister.VAL, generator=False, image_type=DataRegister.PATH, set_task=SEG_CLS)[0]

    def train_data_augment(self, ret):
        ret.update(self.rand_aug1(**ret))
        ret.update(self.rand_aug2(**ret))
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))
        ret.update(self.post_aug(**ret))

        pix_image = ret['pix_image']
        pix_image = self.rand_aug1.apply_image(pix_image, ret)
        pix_image = self.pix_aug.apply_image(pix_image, ret)
        ret['pix_image'] = pix_image
        return ret

    def val_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))
        ret.update(self.post_aug(**ret))
        pix_image = ret['pix_image']
        pix_image = self.pix_aug.apply_image(pix_image, ret)
        ret['pix_image'] = pix_image
        return ret

    def val_data_restore(self, ret):
        ret = self.pix_aug.restore(ret)
        return ret


class FCN_Voc(SegProcess, Voc):
    """
    Usage:
        .. code-block:: python

            from examples.semantic_segment import FCN_Voc as Process

            Process().run(max_epoch=1000)
            {'score': 0.1429}
    """
    model_version = 'FCN'

    def set_model(self):
        from models.semantic_segmentation.FCN import Model
        self.model = Model(
            in_ch=self.in_ch,
            input_size=self.input_size,
            out_features=self.out_features
        )


class Unet_Voc(SegProcess, Voc):
    """
    Usage:
        .. code-block:: python

            from examples.semantic_segment import Unet_Voc as Process

            Process().run(max_epoch=1000)
            {'score': 0.1973}
    """
    model_version = 'Unet'

    def set_model(self):
        from models.semantic_segmentation.Unet import Model
        self.model = Model(
            in_ch=self.in_ch,
            input_size=self.input_size,
            out_features=self.out_features
        )


class DeeplabV3_Voc(SegProcess, Voc):
    """
    Usage:
        .. code-block:: python

            from examples.semantic_segment import DeeplabV3_Voc as Process

            Process().run(max_epoch=1000)
            {'score': 0.3021}
    """
    model_version = 'DeeplabV3'

    def set_model(self):
        from models.semantic_segmentation.DeeplabV3 import Model
        self.model = Model(
            in_ch=self.in_ch,
            input_size=self.input_size,
            out_features=self.out_features
        )
