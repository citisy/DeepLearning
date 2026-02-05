from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from torch import optim

from data_parse.cv_data_parse.data_augmentation import scale, geometry, channel, RandomApply, Apply, pixel_perturbation
from data_parse.cv_data_parse.datasets.base import DataVisualizer, DataRegister
from processor import Process, DataHooks, bundled
from utils import os_lib


def _load_images(images, b, start_idx, end_idx):
    if images:
        if not isinstance(images, (list, tuple)):
            # base on one image
            images = [images for _ in range(b)]
        else:
            images = images[start_idx: end_idx]
        images = [os_lib.loader.load_img(image) if isinstance(image, str) else image for image in images]
    else:
        images = [None] * b

    return images


class ClsProcess(Process):
    is_multi_label = False

    def get_model_inputs(self, loop_inputs, train=True):
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
        images = torch.stack(images)
        # images = images / 255
        inputs = dict(x=images)

        if train:
            _class = [torch.tensor(ret['_class']).to(self.device) for ret in loop_inputs]
            _class = torch.stack(_class)
            inputs.update(true_label=_class)

        return inputs

    def on_train_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs)
        output = self.model(**inputs)
        return output

    def metric(self, **predict_kwargs):
        from metrics import classification
        process_results = self.predict(**predict_kwargs)

        metric_results = {}
        for name, results in process_results.items():
            trues = np.array(results['trues'])
            preds = np.array(results['preds'])
            result = classification.top_metric.f_measure(trues, preds)

            result.update(
                score=result['f']
            )

            metric_results[name] = result

        return metric_results

    def on_val_step(self, loop_objs, top_k=1, thresh=0, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs, train=False)
        model_results = {}
        for name, model in self.models.items():
            outputs = model(**inputs)
            model_pred = outputs['pred']
            if self.is_multi_label:
                preds = []
                argsort = model_pred.argsort(descending=True)
                for arg in argsort:
                    keep = arg[:top_k].cpu().numpy().tolist()
                    preds.append(keep)
            else:
                preds = model_pred.argmax(1).cpu().numpy().tolist()

            model_results[name] = dict(
                model_pred=model_pred,
                preds=preds,
            )

        return model_results

    def on_val_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        loop_inputs = loop_objs['loop_inputs']
        for name, results in model_results.items():
            r = process_results.setdefault(name, dict())
            r.setdefault('trues', []).extend([ret['_class'] for ret in loop_inputs])
            r.setdefault('preds', []).extend(results['preds'])

    def visualize(self, loop_objs, n, **kwargs):
        model_results = loop_objs['model_results']
        loop_inputs = loop_objs['loop_inputs']

        for name, results in model_results.items():
            vis_rets = []
            for i in range(min(n, len(loop_inputs))):
                ret = loop_inputs[i]
                _p = results['preds'][i]
                _id = Path(ret['_id'])
                vis_rets.append(dict(
                    _id=f'({ret["_class"]}_{_p})_{_id.name}',
                    image=ret['ori_image']
                ))

            DataVisualizer(f'{self.cache_dir}/{loop_objs["epoch"]}/{name}', verbose=False, pbar=False)(vis_rets)
            self.get_log_trace(bundled.WANDB).setdefault(f'val_image/{name}', []).extend(
                [self.wandb.Image(cv2.cvtColor(ret['image'], cv2.COLOR_BGR2RGB), caption=ret['_id']) for ret in vis_rets]
            )

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, **kwargs) -> List[dict]:
        images = objs[0][start_idx: end_idx]
        b = len(images)

        images = _load_images(images, b, start_idx, end_idx)

        rets = []
        for image in images:
            rets.append(dict(image=image))

        return rets

    def on_predict_reprocess(self, loop_objs, process_results=dict(), return_keys=('preds',), **kwargs):
        model_results = loop_objs['model_results']
        ret = process_results.setdefault(self.model_name, {})
        preds = ret.setdefault('preds', [])
        _preds = model_results[self.model_name]['preds']
        preds.extend(_preds)
        if hasattr(self, 'classes'):
            classes = ret.setdefault('classes', [])
            if self.is_multi_label:
                _classes = [[self.classes[p] for p in pred] for pred in _preds]
            else:
                _classes = [self.classes[pred] for pred in _preds]
            classes.extend(_classes)


class Mnist(DataHooks):
    dataset_version = 'mnist'
    data_dir = 'data/mnist'

    in_ch = 1
    input_size = 28
    out_features = 10

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.cv_data_parse.datasets.Mnist import Loader

        loader = Loader(self.data_dir)
        if train:
            return loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False)[0]
        else:
            return loader(set_type=DataRegister.TEST, image_type=DataRegister.ARRAY, generator=False)[0]

    train_aug = Apply([
        RandomApply([
            geometry.HFlip(),
            geometry.VFlip(),
        ]),
    ])
    post_aug = Apply([
        pixel_perturbation.MinMax(),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret, train=True) -> dict:
        if train:
            ret.update(self.train_aug(**ret))

        ret.update(self.post_aug(**ret))
        return ret


class Cifar(DataHooks):
    dataset_version = 'cifar-10-batches-py'
    data_dir = 'data/cifar-10-batches-py'

    in_ch = 3
    input_size = 32
    out_features = 10

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.cv_data_parse.datasets.Cifar import Loader

        loader = Loader(self.data_dir)
        if train:
            return loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False)[0]
        else:
            return loader(set_type=DataRegister.TEST, image_type=DataRegister.ARRAY, generator=False)[0]


class ImageNet(DataHooks):
    dataset_version = 'ImageNet2012'
    data_dir = 'data/ImageNet2012'

    in_ch = 3
    input_size = 224
    out_features = 2

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.cv_data_parse.datasets.ImageNet import Loader

        convert_class = {7: 0, 40: 1}

        if train:
            def convert_func(ret):
                ret['_class'] = convert_class[ret['_class']]
                return ret

            loader = Loader(self.data_dir)
            loader.on_end_convert = convert_func

            return loader(
                set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False,
                wnid=[
                    'n02124075',  # Egyptian cat,
                    'n02110341'  # dalmatian, coach dog, carriage dog
                ]
            )[0]

        else:
            def convert_func(ret):
                if ret['_class'] in convert_class:
                    ret['_class'] = convert_class[ret['_class']]
                return ret

            def filter_func(ret):
                if ret['_class'] in [7, 40]:
                    return True

            loader = Loader(self.data_dir)
            loader.on_end_filter = filter_func
            iter_data = loader.load(set_type=DataRegister.VAL, image_type=DataRegister.PATH, generator=False)[0]
            iter_data = list(map(convert_func, iter_data))

            return iter_data

    train_aug = Apply([
        RandomApply([
            geometry.HFlip(),
            geometry.VFlip(),
        ]),
        scale.RuderJitter((256, 257)),
    ])

    val_aug = scale.LetterBox()

    post_aug = Apply([
        channel.Keep3Dims(),
        channel.Keep3Channels(),
        channel.BGR2RGB(),
        pixel_perturbation.MinMax(),
        pixel_perturbation.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        channel.HWC2CHW()
    ])

    def train_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(self.train_aug(**ret))
        ret.update(self.post_aug(**ret))
        return ret

    def val_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(self.val_aug(**ret))
        ret.update(self.post_aug(**ret))
        return ret


class LeNet_mnist(ClsProcess, Mnist):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import LeNet_mnist as Process

            Process().run(max_epoch=10, train_batch_size=256, predict_batch_size=256)
            {'score': 0.9899}
    """
    model_version = 'LeNet'

    def set_model(self):
        from models.image_classification.LeNet import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)


class LeNet_cifar(ClsProcess, Cifar):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import LeNet_cifar as Process

            Process().run(max_epoch=150, train_batch_size=128, predict_batch_size=256)
            {'score': 0.6082}
    """
    model_version = 'LeNet'

    def set_model(self):
        from models.image_classification.LeNet import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)


class AlexNet_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import AlexNet_ImageNet as Process

            Process().run()
            {'p': 0.8461538461538461, 'r': 0.88, 'f': 0.8627450980392156}
    """
    model_version = 'AlexNet'

    def set_model(self):
        from models.image_classification.AlexNet import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)


class Vgg_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import Vgg_ImageNet as Process

            Process().run()
            {'p': 0.9230769230769231, 'r': 0.96, 'f': 0.9411764705882353, 'score': 0.9411764705882353}
    """
    model_version = 'Vgg'

    def set_model(self):
        from models.image_classification.VGG import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)

    def train_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(scale.Jitter((256, 384))(**ret))
        ret.update(RandomApply([geometry.HFlip()])(**ret))
        ret.update(Apply([
            # pixel_perturbation.MinMax(),
            channel.HWC2CHW()
        ])(**ret))
        return ret


class InceptionV1_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import InceptionV1_ImageNet as Process

            Process().run()
            {'p': 0.8363636363636363, 'r': 0.92, 'f': 0.8761904761904761, 'score': 0.8761904761904761}
    """
    model_version = 'InceptionV1'

    def set_model(self):
        from models.image_classification.InceptionV1 import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)


class InceptionV3_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import InceptionV3_ImageNet as Process

            Process().run()
            {'p': 0.98, 'r': 0.98, 'f': 0.98, 'score': 0.98}
    """
    model_version = 'InceptionV3'
    input_size = 299  # special input_size from paper

    def set_model(self):
        from models.image_classification.InceptionV3 import InceptionV3 as Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)


class ResNet_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import ResNet_ImageNet as Process

            Process().run()
            {'p': 0.9230769230769231, 'r': 0.96, 'f': 0.9411764705882353, 'score': 0.9411764705882353}
    """
    model_version = 'ResNet'

    def set_model(self):
        from models.image_classification.ResNet import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)

    # see `torchvision.transforms._presets.ImageClassification`
    train_aug = Apply([
        scale.Jitter((256, 384)),
        RandomApply([
            geometry.HFlip()
        ])
    ])

    post_aug = Apply([
        pixel_perturbation.MinMax(),
        pixel_perturbation.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        channel.HWC2CHW()
    ])

    # see `keras.applications.resnet.preprocess_input`
    # post_aug = Apply([
    #     pixel_perturbation.Normalize(
    #         mean=[103.939, 116.779, 123.68],
    #         std=[1, 1, 1]
    #     ),
    #     channel.HWC2CHW()
    # ])

    def train_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(self.train_aug(**ret))
        ret.update(self.post_aug(**ret))
        return ret


class DenseNet_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import DenseNet_ImageNet as Process

            Process().run()
            {'p': 0.819672131147541, 'r': 1.0, 'f': 0.9009009009009009, 'score': 0.9009009009009009}
    """
    model_version = 'DenseNet'

    def set_model(self):
        from models.image_classification.DenseNet import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)


class SENet_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import SENet_ImageNet as Process

            Process().run()
            {'p': 0.847457627118644, 'r': 1.0, 'f': 0.9174311926605504, 'score': 0.9174311926605504}
    """
    model_version = 'SENet'

    def set_model(self):
        from models.image_classification.SEInception import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)


class SqueezeNet_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import SqueezeNet_ImageNet as Process

            Process().run(train_batch_size=32, predict_batch_size=32)
            {'p': 0.7538461538461538, 'r': 0.98, 'f': 0.8521739130434782, 'score': 0.8521739130434782}
    """
    model_version = 'SqueezeNet'

    def set_model(self):
        from models.image_classification.SqueezeNet import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)


class MobileNetV1_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import MobileNetV1_ImageNet as Process

            Process().run(train_batch_size=32, predict_batch_size=32)
            {'p': 0.9795918367346939, 'r': 0.96, 'f': 0.9696969696969697, 'score': 0.9696969696969697}
    """
    model_version = 'MobileNetV1'

    def set_model(self):
        from models.image_classification.MobileNetV1 import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)


class MobileNetV3_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import MobileNetV3_ImageNet as Process

            Process().run(train_batch_size=32, predict_batch_size=32)
    """
    model_version = 'MobileNetV1'

    def set_model(self):
        from models.image_classification.MobileNetV3 import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)


class ShuffleNetV1_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import ShuffleNetV1_ImageNet as Process

            Process().run(train_batch_size=64, predict_batch_size=64)
            {'p': 0.8679245283018868, 'r': 0.92, 'f': 0.8932038834951457, 'score': 0.8932038834951457}
    """
    model_version = 'ShuffleNet'

    def set_model(self):
        from models.image_classification.ShuffleNetV1 import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)


class IGC_cifar(ClsProcess, Cifar):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import IGC_cifar as Process

            Process().run(train_batch_size=64, predict_batch_size=64)
            {'score': 0.8058}
    """
    model_version = 'IGC'

    def set_model(self):
        from models.image_classification.IGCV1 import Model
        from models.layers import SimpleInModule

        self.model = Model(in_module=SimpleInModule(out_channels=self.in_ch), out_features=self.out_features)


class CondenseNet_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from bundles.image_classification import CondenseNet_ImageNet as Process

            Process().run(train_batch_size=64, predict_batch_size=64)
            {'p': 0.9333333333333333, 'r': 0.84, 'f': 0.8842105263157894, 'score': 0.8842105263157894}
    """
    model_version = 'CondenseNet'

    def set_model(self):
        from models.image_classification.CondenseNet import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)


class ViT_ImageNet(ClsProcess, ImageNet):
    """note, work better on a large dataset

    Usage:
        .. code-block:: python

            from bundles.image_classification import ViT_ImageNet as Process

            Process().run(max_epoch=300, train_batch_size=32, predict_batch_size=32)
            {'p': 0.7049180212308521, 'r': 0.86, 'f': 0.7747742727054547, 'score': 0.7747742727054547}
    """
    model_version = 'ViT'
    config_version = 'B_16'

    def set_model(self):
        from models.image_classification.ViT import Model, Config
        self.model = Model(
            in_ch=self.in_ch,
            input_size=self.input_size,
            out_features=self.out_features,
            **Config.get(self.config_version)
        )

    def set_optimizer(self, lr=0.001, momentum=0.9, weight_decay=1e-4, **kwargs):
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


class Vit_ImageNet_Pretrained(ViT_ImageNet):
    """pretrained_model from https://modelscope.cn/models/iic/cv_vit-base_image-classification_Dailylife-labels/summary

    Usage:
        .. code-block:: python

            from bundles.image_classification import Vit_ImageNet_Pretrained as Process

            process = Process(
                pretrained_model='xxx/pytorch_model.pt'
            )
            process.init()

            process.single_predict(
                'xxx.png',
                top_k=5
            )
    """
    out_features = 1296
    config_version = 'B_16_H_12'
    is_multi_label = True

    val_aug = scale.RuderLetterBox(
        mid_dst=256,
        interpolation=3
    )

    post_aug = Apply([
        channel.Keep3Dims(),
        channel.Keep3Channels(),
        channel.BGR2RGB(),
        pixel_perturbation.Normalize(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
        ),
        channel.HWC2CHW()
    ])

    def load_pretrained(self):
        from models.image_classification.ViT import WeightConverter
        from utils import torch_utils

        tensor = torch_utils.Load.from_ckpt(self.pretrained_model)

        state_dict = tensor['state_dict']
        meta = tensor['meta']

        self.classes = meta['CLASSES']

        state_dict = WeightConverter.from_modelscope(state_dict)
        self.model.load_state_dict(state_dict, strict=True)
        self.log('Loaded pretrained success!')
