import cv2
import torch
from torch import optim
import numpy as np
from pathlib import Path
from utils import torch_utils
from processor import Process, DataHooks, bundled
from data_parse.cv_data_parse.base import DataVisualizer, DataRegister
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, channel, RandomApply, Apply


class ClsProcess(Process):
    def on_train_step(self, rets, container, **kwargs) -> dict:
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        _class = [torch.tensor(ret['_class']).to(self.device) for ret in rets]
        images = torch.stack(images)
        _class = torch.stack(_class)
        images = images / 255
        output = self.model(images, _class)

        return output

    def metric(self, **predict_kwargs):
        from metrics import classification
        container = self.predict(**predict_kwargs)

        metric_results = {}
        for name, results in container['model_results'].items():
            trues = np.array(results['trues'])
            preds = np.array(results['preds'])
            result = classification.top_metric.f_measure(trues, preds)

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

            model_results[name] = dict(
                outputs=outputs['pred'],
                preds=outputs['pred'].argmax(1).cpu().numpy().tolist(),
            )

        return model_results

    def on_val_reprocess(self, rets, model_results, container, **kwargs):
        for name, results in model_results.items():
            r = container['model_results'].setdefault(name, dict())
            r.setdefault('trues', []).extend([ret['_class'] for ret in rets])
            r.setdefault('preds', []).extend(results['preds'])

    def visualize(self, rets, model_results, n, **kwargs):
        for name, results in model_results.items():
            vis_rets = []
            for i in range(n):
                ret = rets[i]
                _p = results['preds'][i]
                _id = Path(ret['_id'])
                vis_rets.append(dict(
                    _id=f'{_id.stem}({ret["_class"]}_{_p}){_id.suffix}',
                    image=ret['ori_image']
                ))

            DataVisualizer(f'{self.cache_dir}/{self.counters["epoch"]}/{name}', verbose=False, pbar=False)(vis_rets)
            self.get_log_trace(bundled.WANDB).setdefault(f'val_image/{name}', []).extend(
                [self.wandb.Image(cv2.cvtColor(ret['image'], cv2.COLOR_BGR2RGB), caption=ret['_id']) for ret in vis_rets]
            )


class Mnist(DataHooks):
    dataset_version = 'mnist'
    data_dir = 'data/mnist'

    in_ch = 1
    input_size = 28
    out_features = 10

    def get_train_data(self):
        from data_parse.cv_data_parse.Mnist import Loader

        loader = Loader(self.data_dir)
        return loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False)[0]

    def get_val_data(self):
        from data_parse.cv_data_parse.Mnist import Loader

        loader = Loader(self.data_dir)
        return loader(set_type=DataRegister.TEST, image_type=DataRegister.ARRAY, generator=False)[0]

    aug = Apply([
        RandomApply([
            geometry.HFlip(),
            geometry.VFlip(),
        ]),
    ])
    post_aug = channel.HWC2CHW()

    def train_data_augment(self, ret):
        ret.update(self.aug(**ret))
        ret.update(self.post_aug(**ret))
        return ret

    def val_data_augment(self, ret):
        ret.update(self.post_aug(**ret))
        return ret


class Cifar(DataHooks):
    dataset_version = 'cifar-10-batches-py'
    data_dir = 'data/cifar-10-batches-py'

    in_ch = 3
    input_size = 32
    out_features = 10

    def get_train_data(self):
        from data_parse.cv_data_parse.Cifar import Loader

        loader = Loader(self.data_dir)
        return loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False)[0]

    def get_val_data(self):
        from data_parse.cv_data_parse.Cifar import Loader

        loader = Loader(self.data_dir)
        return loader(set_type=DataRegister.TEST, image_type=DataRegister.ARRAY, generator=False)[0]


class ImageNet(DataHooks):
    dataset_version = 'ImageNet2012'
    data_dir = 'data/ImageNet2012'

    in_ch = 3
    input_size = 224
    out_features = 2

    def get_train_data(self):
        from data_parse.cv_data_parse.ImageNet import Loader

        convert_class = {7: 0, 40: 1}

        def convert_func(ret):
            ret['_class'] = convert_class[ret['_class']]
            return ret

        loader = Loader(self.data_dir)
        loader.on_end_convert = convert_func

        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False,
                      wnid=[
                          'n02124075',  # Egyptian cat,
                          'n02110341'  # dalmatian, coach dog, carriage dog
                      ]
                      )[0]

        return data

    def get_val_data(self):
        from data_parse.cv_data_parse.ImageNet import Loader

        convert_class = {7: 0, 40: 1}

        def convert_func(ret):
            if ret['_class'] in convert_class:
                ret['_class'] = convert_class[ret['_class']]
            return ret

        def filter_func(ret):
            if ret['_class'] in [7, 40]:
                return True

        loader = Loader(self.data_dir)
        loader.on_end_filter = filter_func

        data = loader(set_type=DataRegister.VAL, image_type=DataRegister.PATH, generator=False)[0]
        data = list(map(convert_func, data))

        return data

    def train_data_augment(self, ret):
        ret.update(scale.Proportion()(**ret, dst=256))
        ret.update(dst=self.input_size)
        ret.update(Apply([
            crop.Random(),
            RandomApply([
                geometry.HFlip(),
                geometry.VFlip(),
            ]),
            # pixel_perturbation.MinMax(),
            channel.HWC2CHW()
        ])(**ret))
        return ret

    def val_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(Apply([
            scale.LetterBox(),
            channel.HWC2CHW()
        ])(**ret))
        return ret


class LeNet_mnist(ClsProcess, Mnist):
    """
    Usage:
        .. code-block:: python

            from examples.image_classification import LeNet_mnist as Process

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

            from examples.image_classification import LeNet_cifar as Process

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

            from examples.image_classification import AlexNet_ImageNet as Process

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

            from examples.image_classification import Vgg_ImageNet as Process

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

            from examples.image_classification import InceptionV1_ImageNet as Process

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

            from examples.image_classification import InceptionV3_ImageNet as Process

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

            from examples.image_classification import ResNet_ImageNet as Process

            Process().run()
            {'p': 0.9230769230769231, 'r': 0.96, 'f': 0.9411764705882353, 'score': 0.9411764705882353}
    """
    model_version = 'ResNet'

    def set_model(self):
        from models.image_classification.ResNet import Model, Res50_config
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


class DenseNet_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from examples.image_classification import DenseNet_ImageNet as Process

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

            from examples.image_classification import SENet_ImageNet as Process

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

            from examples.image_classification import SqueezeNet_ImageNet as Process

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

            from examples.image_classification import MobileNetV1_ImageNet as Process

            Process().run(train_batch_size=32, predict_batch_size=32)
            {'p': 0.9795918367346939, 'r': 0.96, 'f': 0.9696969696969697, 'score': 0.9696969696969697}
    """
    model_version = 'MobileNetV1'

    def set_model(self):
        from models.image_classification.MobileNetV1 import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)


class ShuffleNetV1_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from examples.image_classification import ShuffleNetV1_ImageNet as Process

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

            from examples.image_classification import IGC_cifar as Process

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

            from examples.image_classification import CondenseNet_ImageNet as Process

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

            from examples.image_classification import ViT_ImageNet as Process

            Process().run(max_epoch=300, train_batch_size=32, predict_batch_size=32)
            {'p': 0.7049180212308521, 'r': 0.86, 'f': 0.7747742727054547, 'score': 0.7747742727054547}
    """
    model_version = 'ViT'

    def set_model(self):
        from models.image_classification.ViT import Model
        self.model = Model(self.in_ch, self.input_size, self.out_features)

    def set_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
