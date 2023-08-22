import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.layers import SimpleInModule
from utils.os_lib import MemoryInfo
from utils import configs
from metrics import classifier
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, channel, RandomApply, Apply
from data_parse import DataRegister
from data_parse.cv_data_parse.base import DataVisualizer
from .base import Process, BaseDataset
from pathlib import Path


class ClsProcess(Process):
    dataset = BaseDataset

    def fit(self, max_epoch=100, batch_size=16, save_period=None, metric_kwargs=dict(), **dataloader_kwargs):
        train_dataloader, val_dataloader, metric_kwargs = self.on_train_start(batch_size, metric_kwargs, **dataloader_kwargs)

        for i in range(self.start_epoch, max_epoch):
            self.model.train()
            pbar = tqdm(train_dataloader, desc=f'train {i}/{max_epoch}')
            total_loss = 0
            total_batch = 0
            mean_loss = 0

            for rets in pbar:
                images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                _class = [torch.tensor(ret['_class']).to(self.device) for ret in rets]
                images = torch.stack(images)
                _class = torch.stack(_class)
                images = images / 255

                self.optimizer.zero_grad()

                output = self.model(images, _class)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_batch += len(rets)

                mean_loss = total_loss / total_batch
                pbar.set_postfix({
                    'loss': f'{loss.item():.06}',
                    'mean_loss': f'{mean_loss:.06}',
                    # 'cpu_info': MemoryInfo.get_process_mem_info(),
                    # 'gpu_info': MemoryInfo.get_gpu_mem_info()
                })

            if self.on_train_epoch_end(i, save_period, mean_loss, val_dataloader, **metric_kwargs):
                break

        self.wandb.finish()

    def metric(self, *args, **predict_kwargs):
        true, pred = self.predict(*args, **predict_kwargs)
        result = classifier.top_metric.f_measure(true, pred)

        result.update(
            score=result['f']
        )

        return result

    def predict(self, val_dataloader=None, batch_size=128, cur_epoch=-1, model=None, visualize=False, max_vis_num=float('inf'), save_ret_func=None, **dataloader_kwargs):
        if val_dataloader is None:
            val_dataloader = self.on_val_start(batch_size, **dataloader_kwargs)

        model = model or self.model
        model.to(self.device)

        pred = []
        true = []
        vis_num = 0

        with torch.no_grad():
            self.model.eval()
            for rets in tqdm(val_dataloader):
                images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images = torch.stack(images)
                images = images / 255

                outputs = self.model(images)
                outputs['pred'] = outputs['pred'].argmax(1).cpu().detach().numpy()
                pred.extend(outputs['pred'].tolist())
                true.extend([ret['_class'] for ret in rets])

                vis_num = self.on_val_step_end(rets, outputs, cur_epoch, visualize, batch_size, max_vis_num, vis_num)

        pred = np.array(pred)
        true = np.array(true)

        return true, pred

    def on_val_step_end(self, rets, outputs, cur_epoch, visualize, batch_size, max_vis_num, vis_num):
        if visualize:
            n = min(batch_size, max_vis_num - vis_num)
            if n > 0:
                for ret, _p in zip(rets, outputs['pred']):
                    _id = Path(ret['_id'])
                    ret['_id'] = f'{_id.stem}(t={ret["_class"]},p={_p}){_id.suffix}'
                    ret['image'] = ret['ori_image']
                DataVisualizer(f'{self.save_result_dir}/{cur_epoch}', verbose=False, pbar=False)(rets[:n])
                self.log_info.setdefault('val_image', []).extend([self.wandb.Image(ret['image'], caption=ret['_id']) for ret in rets[:n]])
                vis_num += n

        return vis_num


class Mnist(Process):
    def get_train_data(self):
        from data_parse.cv_data_parse.Mnist import Loader

        loader = Loader('data/mnist')
        return loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False)[0]

    def get_val_data(self):
        from data_parse.cv_data_parse.Mnist import Loader

        loader = Loader('data/mnist')
        return loader(set_type=DataRegister.TEST, image_type=DataRegister.ARRAY, generator=False)[0]

    def data_augment(self, ret):
        ret.update(Apply([
            RandomApply([
                geometry.HFlip(),
                geometry.VFlip(),
            ]),
            channel.HWC2CHW()
        ])(**ret))
        return ret

    def val_data_augment(self, ret):
        ret.update(Apply([
            channel.HWC2CHW()
        ])(**ret))
        return ret


class Cifar(Process):
    def get_train_data(self):
        from data_parse.cv_data_parse.Cifar import Loader

        loader = Loader('data/cifar-10-batches-py')
        return loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False)[0]

    def get_val_data(self):
        from data_parse.cv_data_parse.Cifar import Loader

        loader = Loader('data/cifar-10-batches-py')
        return loader(set_type=DataRegister.TEST, image_type=DataRegister.ARRAY, generator=False)[0]


class ImageNet(Process):
    def data_augment(self, ret):
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

    def get_train_data(self):
        from data_parse.cv_data_parse.ImageNet import Loader

        convert_class = {7: 0, 40: 1}

        def convert_func(ret):
            ret['_class'] = convert_class[ret['_class']]
            return ret

        loader = Loader(f'data/ImageNet2012')
        loader.convert_func = convert_func

        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False,
                      wnid=[
                          'n02124075',  # Egyptian cat,
                          'n02110341'  # dalmatian, coach dog, carriage dog
                      ]
                      )[0]

        return data

    def val_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(Apply([
            scale.LetterBox(),
            channel.HWC2CHW()
        ])(**ret))
        return ret

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

        loader = Loader(f'data/ImageNet2012')
        loader.filter_func = filter_func

        data = loader(set_type=DataRegister.VAL, image_type=DataRegister.PATH, generator=False)[0]
        data = list(map(convert_func, data))

        return data


class LeNet_mnist(ClsProcess, Mnist):
    """
    Usage:
        .. code-block:: python

            from examples.image_cls import LeNet_mnist as Process

            Process().run(max_epoch=10, train_batch_size=256, predict_batch_size=256)
            {'score': 0.9899}
    """

    def __init__(self,
                 in_ch=1,
                 input_size=28,
                 out_features=10,
                 model_version='LeNet',
                 dataset_version='mnist',
                 **kwargs
                 ):
        from models.image_classifier.LeNet import Model

        super().__init__(
            model=Model(in_ch, input_size, out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            **kwargs
        )


class LeNet_cifar(ClsProcess, Cifar):
    """
    Usage:
        .. code-block:: python

            from examples.image_cls import LeNet_cifar as Process

            Process().run(max_epoch=150, train_batch_size=128, predict_batch_size=256)
            {'score': 0.6082}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=32,
                 out_features=10,
                 model_version='LeNet',
                 dataset_version='cifar-10-batches-py',
                 **kwargs
                 ):
        from models.image_classifier.LeNet import Model

        super().__init__(
            model=Model(in_ch, input_size, out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )


class AlexNet_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from examples.image_cls import AlexNet_ImageNet as Process

            Process().run()
            {'p': 0.8461538461538461, 'r': 0.88, 'f': 0.8627450980392156}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=224,
                 out_features=2,
                 model_version='AlexNet',
                 dataset_version='ImageNet2012',
                 **kwargs
                 ):
        from models.image_classifier.AlexNet import Model

        super().__init__(
            model=Model(in_ch, input_size, out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )


class Vgg_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from examples.image_cls import Vgg_ImageNet as Process

            Process().run()
            {'p': 0.9230769230769231, 'r': 0.96, 'f': 0.9411764705882353, 'score': 0.9411764705882353}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=224,
                 out_features=2,
                 model_version='Vgg',
                 dataset_version='ImageNet2012',
                 **kwargs
                 ):
        from models.image_classifier.VGG import Model

        super().__init__(
            model=Model(in_ch, input_size, out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )

    def data_augment(self, ret):
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

            from examples.image_cls import InceptionV1_ImageNet as Process

            Process().run()
            {'p': 0.8363636363636363, 'r': 0.92, 'f': 0.8761904761904761, 'score': 0.8761904761904761}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=224,
                 out_features=2,
                 model_version='InceptionV1',
                 dataset_version='ImageNet2012',
                 **kwargs
                 ):
        from models.image_classifier.InceptionV1 import Model

        super().__init__(
            model=Model(in_ch, input_size, out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )


class InceptionV3_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from examples.image_cls import InceptionV3_ImageNet as Process

            Process().run()
            {'p': 0.98, 'r': 0.98, 'f': 0.98, 'score': 0.98}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=299,
                 out_features=2,
                 model_version='InceptionV3',
                 dataset_version='ImageNet2012',
                 **kwargs
                 ):
        from models.image_classifier.InceptionV3 import InceptionV3 as Model

        super().__init__(
            model=Model(in_ch, input_size, out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )


class ResNet_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from examples.image_cls import ResNet_ImageNet as Process

            Process().run()
            {'p': 0.9230769230769231, 'r': 0.96, 'f': 0.9411764705882353, 'score': 0.9411764705882353}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=224,
                 out_features=2,
                 model_version='ResNet',
                 dataset_version='ImageNet2012',
                 **kwargs
                 ):
        from models.image_classifier.ResNet import Model, Res50_config

        super().__init__(
            model=Model(in_ch, input_size, out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )

    def data_augment(self, ret):
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

            from examples.image_cls import DenseNet_ImageNet as Process

            Process().run()
            {'p': 0.819672131147541, 'r': 1.0, 'f': 0.9009009009009009, 'score': 0.9009009009009009}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=224,
                 out_features=2,
                 model_version='DenseNet',
                 dataset_version='ImageNet2012',
                 **kwargs
                 ):
        from models.image_classifier.DenseNet import Model

        super().__init__(
            model=Model(in_ch, input_size, out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )


class SENet_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from examples.image_cls import SENet_ImageNet as Process

            Process().run()
            {'p': 0.847457627118644, 'r': 1.0, 'f': 0.9174311926605504, 'score': 0.9174311926605504}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=224,
                 out_features=2,
                 model_version='SENet',
                 dataset_version='ImageNet2012',
                 **kwargs
                 ):
        from models.image_classifier.SEInception import Model

        super().__init__(
            model=Model(in_ch, input_size, out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )


class SqueezeNet_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from examples.image_cls import SqueezeNet_ImageNet as Process

            Process().run(train_batch_size=32, predict_batch_size=32)
            {'p': 0.7538461538461538, 'r': 0.98, 'f': 0.8521739130434782, 'score': 0.8521739130434782}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=224,
                 out_features=2,
                 model_version='SqueezeNet',
                 dataset_version='ImageNet2012',
                 **kwargs
                 ):
        from models.image_classifier.SqueezeNet import Model

        super().__init__(
            model=Model(in_ch, input_size, out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )


class MobileNetV1_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from examples.image_cls import MobileNetV1_ImageNet as Process

            Process().run(train_batch_size=32, predict_batch_size=32)
            {'p': 0.9795918367346939, 'r': 0.96, 'f': 0.9696969696969697, 'score': 0.9696969696969697}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=224,
                 out_features=2,
                 model_version='MobileNetV1',
                 dataset_version='ImageNet2012',
                 **kwargs
                 ):
        from models.image_classifier.MobileNetV1 import Model

        super().__init__(
            model=Model(in_ch, input_size, out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )


class ShuffleNetV1_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from examples.image_cls import ShuffleNetV1_ImageNet as Process

            Process().run(train_batch_size=64, predict_batch_size=64)
            {'p': 0.8679245283018868, 'r': 0.92, 'f': 0.8932038834951457, 'score': 0.8932038834951457}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=224,
                 out_features=2,
                 model_version='ShuffleNet',
                 dataset_version='ImageNet2012',
                 **kwargs
                 ):
        from models.image_classifier.ShuffleNetV1 import Model

        super().__init__(
            model=Model(in_ch, input_size, out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )


class IGC_cifar(ClsProcess, Cifar):
    """
    Usage:
        .. code-block:: python

            from examples.image_cls import IGC_cifar as Process

            Process().run(train_batch_size=64, predict_batch_size=64)
            {'score': 0.8058}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=32,
                 out_features=10,
                 model_version='IGC',
                 dataset_version='cifar-10-batches-py',
                 **kwargs
                 ):
        from models.image_classifier.IGCV1 import Model

        super().__init__(
            model=Model(in_module=SimpleInModule(out_channels=in_ch), out_features=out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )

        self.input_size = input_size


class CondenseNet_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from examples.image_cls import CondenseNet_ImageNet as Process

            Process().run(train_batch_size=64, predict_batch_size=64)
            {'p': 0.9333333333333333, 'r': 0.84, 'f': 0.8842105263157894, 'score': 0.8842105263157894}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=224,
                 out_features=2,
                 model_version='CondenseNet',
                 dataset_version='ImageNet2012',
                 **kwargs
                 ):
        from models.image_classifier.CondenseNet import Model

        super().__init__(
            model=Model(in_ch, input_size, out_features),
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )


class ViT_ImageNet(ClsProcess, ImageNet):
    """
    Usage:
        .. code-block:: python

            from examples.image_cls import ViT_ImageNet as Process

            Process().run(max_epoch=300, train_batch_size=32, predict_batch_size=32)
            {'p': 0.7049180212308521, 'r': 0.86, 'f': 0.7747742727054547, 'score': 0.7747742727054547}
    """

    def __init__(self,
                 in_ch=3,
                 input_size=224,
                 out_features=2,
                 model_version='ViT',
                 dataset_version='ImageNet2012',
                 **kwargs
                 ):
        from models.image_classifier.ViT import Model
        model = Model(in_ch, input_size, out_features)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

        super().__init__(
            model=model,
            optimizer=optimizer,
            model_version=model_version,
            dataset_version=dataset_version,
            input_size=input_size,
            **kwargs
        )
