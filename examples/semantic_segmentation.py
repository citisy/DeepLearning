import copy
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
from torch import optim

from data_parse.cv_data_parse.base import DataVisualizer
from data_parse.cv_data_parse.data_augmentation import scale, geometry, channel, RandomApply, Apply, pixel_perturbation
from processor import Process, DataHooks, bundled, BaseImgDataset
from utils import visualize, configs, torch_utils, os_lib


class SegDataset(BaseImgDataset):
    def process_one(self, idx):
        ret = copy.deepcopy(self.iter_data[idx])
        if isinstance(ret['image'], str):
            ret['image_path'] = ret['image']
            ret['image'] = cv2.imread(ret['image'])
            ret['label_mask_path'] = ret['label_mask']
            # note that, use PIL.Image to get the label image
            ret['label_mask'] = np.asarray(Image.open(ret['label_mask']))

        ret['ori_image'] = ret['image']
        ret['ori_label_mask'] = ret['label_mask']
        ret['idx'] = idx

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret


class SegProcess(Process):
    use_scaler = True
    use_scheduler = True

    out_features: int

    def set_optimizer(self, lr=0.001, momentum=0.9, weight_decay=1e-4, **kwargs):
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    def get_model_inputs(self, rets, train=True):
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        images = torch.stack(images)
        # images = images / 255

        effective_areas = []
        for ret in rets:
            h, w = images.shape[-2:]
            pad_info = ret['crop.Pad']
            t = pad_info.get('t', 0)
            d = pad_info.get('d', 0)
            l = pad_info.get('l', 0)
            r = pad_info.get('r', 0)
            effective_areas.append([l, w - r, t, h - d])

        r = dict(
            x=images,
            effective_areas=effective_areas
        )
        if train:
            label_masks = [torch.from_numpy(ret.pop('label_mask')).to(self.device, non_blocking=True, dtype=torch.long) for ret in rets]
            label_masks = torch.stack(label_masks)
            r.update(label_masks=label_masks)

        return r

    def on_train_step(self, rets, **kwargs) -> dict:
        inputs = self.get_model_inputs(rets)

        with torch.cuda.amp.autocast(True):
            output = self.model(**inputs)

        return output

    def metric(self, *args, **kwargs):
        from metrics import multi_classification

        container = self.predict(*args, **kwargs)

        metric_results = {}
        for name, results in container['model_results'].items():
            # ignore background
            pred = np.concatenate([i.flatten() for i in results['preds']]) - 1
            true = np.concatenate([i.flatten() for i in results['trues']]) - 1

            result = multi_classification.TopMetric(n_class=self.out_features, ignore_class=(-1, 254)).f1(true, pred)

            result.update(
                score=result['f']
            )

            metric_results[name] = result

        return metric_results

    def on_val_step(self, rets, **kwargs) -> dict:
        inputs = self.get_model_inputs(rets, train=False)
        model_results = {}
        for name, model in self.models.items():
            outputs = model(**inputs)
            outputs = outputs.cpu().numpy().astype(np.uint8)

            preds = []
            for (output, ret) in zip(outputs, rets):
                output = configs.ConfigObjParse.merge_dict(ret, {'image': output})
                output = self.val_data_restore(output)
                preds.append(output['image'])

            model_results[name] = dict(
                outputs=outputs,
                preds=preds,
            )

        return model_results

    def on_val_reprocess(self, rets, model_results, **kwargs):
        for name, results in model_results.items():
            r = self.val_container['model_results'].setdefault(name, dict())
            r.setdefault('trues', []).extend([ret['ori_label_mask'] for ret in rets])
            r.setdefault('preds', []).extend(results['preds'])

    def visualize(self, rets, model_results, n, **kwargs):
        for name, results in model_results.items():
            vis_trues = []
            vis_preds = []
            for i in range(n):
                gt_ret = rets[i]
                pred = results['preds'][i]

                true_image = None
                if 'ori_label_mask' in gt_ret:
                    true = gt_ret['ori_label_mask']
                    true_image = np.zeros((*true.shape, 3), dtype=true.dtype) + 255
                    for c in np.unique(true):
                        true_image[true == c] = visualize.get_color_array(c)

                pred_image = np.zeros((*pred.shape, 3), dtype=pred.dtype) + 255
                for c in np.unique(pred):
                    pred_image[pred == c] = visualize.get_color_array(c)

                vis_trues.append(dict(
                    _id=gt_ret['_id'],
                    image=gt_ret['ori_image'],
                    label_mask=true_image
                ))
                vis_preds.append(dict(
                    image=pred_image
                ))

            visualizer = DataVisualizer(f'{self.cache_dir}/{self.counters.get("epoch", "")}/{name}', verbose=False, pbar=False)
            cache_image = visualizer(vis_trues, vis_preds, return_image=True)
            self.get_log_trace(bundled.WANDB).setdefault(f'val_image/{name}', []).extend(
                [self.wandb.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=Path(str(r['_id'])).stem)
                 for img, r in zip(cache_image, vis_trues)]
            )

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, ids=None, **kwargs):
        images = objs[0]
        if ids is None:
            ids = [f'{i}.png' for i in range(start_idx, end_idx)]

        rets = []
        for image, _id in zip(images[start_idx: end_idx], ids):
            if isinstance(image, str):
                image = os_lib.loader.load_img(image)
            rets.append(dict(
                ori_image=image,
                image=image,
                _id=_id
            ))
        return rets

    def on_predict_reprocess(self, rets, model_results, **kwargs):
        for name, results in model_results.items():
            r = self.predict_container['model_results'].setdefault(name, dict())
            r.setdefault('preds', []).extend(results['preds'])

    def on_predict_step_end(self, rets, model_results, **kwargs):
        self.visualize(rets, model_results, len(rets), **kwargs)

    def on_predict_end(self, **kwargs) -> List:
        return self.predict_container['model_results'][self.model_name]['preds']


class SegDataProcess(DataHooks):
    train_dataset_ins = SegDataset
    val_dataset_ins = SegDataset


class Voc(SegDataProcess):
    dataset_version = 'VOC2012'
    data_dir = 'data/VOC2012'
    input_size = 512
    in_ch = 3
    out_features = 20

    aug = Apply([
        # scale.Proportion(choice_type=3),
        # crop.Random(is_pad=False),
        scale.LetterBox(),  # there are gray lines
    ])

    # note that, use cv2.INTER_NEAREST mode to resize
    mask_aug = Apply([
        # scale.Proportion(choice_type=3, interpolation=1),
        # crop.Random(is_pad=False, fill=255),
        scale.LetterBox(interpolation=1, fill=255),  # there are gray lines
    ])

    rand_aug1 = RandomApply([
        geometry.HFlip(),
        geometry.VFlip(),
    ])
    rand_aug2 = RandomApply([pixel_perturbation.Jitter()])

    post_aug = Apply([
        pixel_perturbation.MinMax(),
        # pixel_perturbation.Normalize(127.5, 127.5),
        channel.HWC2CHW()
    ])

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.cv_data_parse.Voc import Loader, DataRegister, SEG_CLS

        loader = Loader(self.data_dir)
        if train:
            return loader(set_type=DataRegister.TRAIN, generator=False, image_type=DataRegister.PATH, set_task=SEG_CLS)[0]
        else:
            return loader(set_type=DataRegister.VAL, generator=False, image_type=DataRegister.PATH, set_task=SEG_CLS)[0]

    def data_augment(self, ret, train=True) -> dict:
        if train:
            ret.update(self.rand_aug1(**ret))
            ret.update(self.rand_aug2(**ret))

        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))
        ret.update(self.post_aug(**ret))

        label_mask = ret['label_mask']
        if train:
            label_mask = self.rand_aug1.apply_image(label_mask, ret)
        label_mask = self.mask_aug.apply_image(label_mask, ret)
        ret['label_mask'] = label_mask

        return ret

    def val_data_restore(self, ret):
        ret = self.mask_aug.restore(ret)
        return ret

    def predict_data_augment(self, ret) -> dict:
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))
        ret.update(self.post_aug(**ret))
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


class VocForSAM(Voc):
    input_size = 1024

    aug = Apply([
        scale.LetterBox(fill=[123.675, 116.28, 103.53], pad_type=(1, 1)),  # there are gray lines
    ])

    # note that, use cv2.INTER_NEAREST mode to resize
    mask_aug = Apply([
        scale.LetterBox(interpolation=1, fill=255),  # there are gray lines
    ])

    rand_aug1 = RandomApply([
        geometry.HFlip(),
        geometry.VFlip(),
    ])
    rand_aug2 = RandomApply([pixel_perturbation.Jitter()])

    post_aug = Apply([
        pixel_perturbation.Normalize(
            [123.675, 116.28, 103.53],
            [58.395, 57.12, 57.375]
        ),
        channel.HWC2CHW()
    ])


class SAM_Voc(SegProcess, VocForSAM):
    """
    Usage:
        .. code-block:: python

            from examples.semantic_segment import DeeplabV3_Voc as Process
    """

    model_version = 'SAM'

    def set_model(self):
        from models.semantic_segmentation.SAM import Model

        self.model = Model(
            in_ch=self.in_ch,
            input_size=self.input_size,
        )

    def load_pretrained(self):
        if hasattr(self, 'pretrain_model'):
            from models.semantic_segmentation.SAM import WeightConverter

            state_dict = torch_utils.Load.from_file(self.pretrain_model)
            state_dict = WeightConverter.from_official(state_dict)
            self.model.load_state_dict(state_dict, strict=False)


class U2net_Voc(SegProcess, Voc):
    model_version = 'U2net'

    input_size = 320

    aug = scale.Rectangle(interpolation=2)

    post_aug = Apply([
        channel.BGR2RGB(),
        pixel_perturbation.MinMax(),
        pixel_perturbation.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        channel.HWC2CHW(),
    ])

    mask_aug = Apply([
        scale.Rectangle(interpolation=2),
        pixel_perturbation.MinMax(),
        channel.Keep3Dims(),
        channel.HWC2CHW(),
    ])

    def set_model(self):
        from models.semantic_segmentation.u2net import Model

        self.model = Model(self.in_ch)
