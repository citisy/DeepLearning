import cv2
import copy
import numpy as np
import torch
from torch import optim, nn
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, channel, RandomApply, Apply, complex, pixel_perturbation
from pathlib import Path
from PIL import Image
from data_parse.cv_data_parse.base import DataVisualizer
from processor import Process, DataHooks, bundled, BaseImgDataset
from utils import visualize, torch_utils, configs


class SegDataset(BaseImgDataset):
    def process_one(self, idx):
        ret = copy.deepcopy(self.iter_data[idx])
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
    use_scaler = True
    use_scheduler = True

    def set_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    def on_train_step(self, rets, container, **kwargs) -> dict:
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        images = torch.stack(images)

        pix_images = [torch.from_numpy(ret.pop('pix_image')).to(self.device, non_blocking=True, dtype=torch.long) for ret in rets]
        pix_images = torch.stack(pix_images)

        images = images / 255

        with torch.cuda.amp.autocast(True):
            output = self.model(images, pix_images)

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

    def on_val_step(self, rets, container, **kwargs) -> dict:
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        images = torch.stack(images)
        images = images / 255

        models = container['models']
        model_results = {}
        for name, model in models.items():
            outputs = model(images)
            outputs = outputs.cpu().numpy().astype(np.uint8)

            preds = []
            for i in range(len(images)):
                output = outputs[i]
                ret = rets[i]

                output = configs.merge_dict(ret, {'image': output})
                output = self.val_data_restore(output)
                preds.append(output['image'])

            model_results[name] = dict(
                outputs=outputs,
                preds=preds,
            )

        return model_results

    def on_val_reprocess(self, rets, model_results, container, **kwargs):
        for name, results in model_results.items():
            r = container['model_results'].setdefault(name, dict())
            r.setdefault('trues', []).extend([ret['ori_pix_image'] for ret in rets])
            r.setdefault('preds', []).extend(results['preds'])

    def visualize(self, rets, model_results, n, **kwargs):
        for ret in rets:
            ret.pop('bboxes')
            ret.pop('pix_image')

        for name, results in model_results.items():
            vis_trues = []
            vis_preds = []
            for i in range(n):
                gt_ret = rets[i]
                pred = results['preds'][i]
                true = gt_ret['ori_pix_image']
                true_image = np.zeros((*true.shape, 3), dtype=true.dtype) + 255
                pred_image = np.zeros((*pred.shape, 3), dtype=pred.dtype) + 255
                for i in range(self.out_features + 1):
                    true_image[true == i] = visualize.get_color_array(i)
                    pred_image[pred == i] = visualize.get_color_array(i)

                vis_trues.append(dict(
                    _id=gt_ret['_id'],
                    image=gt_ret['ori_image'],
                    pix_image=true_image
                ))
                vis_preds.append(dict(
                    image=pred_image
                ))

            cache_image = DataVisualizer(f'{self.cache_dir}/{self.counters["epoch"]}/{name}', verbose=False, pbar=False)(vis_trues, vis_preds, return_image=True)
            self.get_log_trace(bundled.WANDB).setdefault(f'val_image/{name}', []).extend(
                [self.wandb.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=Path(r['_id']).stem) for img, r in zip(cache_image, vis_trues)]
            )


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
    pix_aug = Apply([
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
