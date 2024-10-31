from torch import nn

from utils import torch_utils, math_utils
from . import BaseImgClsModel
from .. import bundles
from ..layers import Conv, Linear, Residual


class Config(bundles.Config):
    # refer to table 1
    # (n_res, out_ch, n_conv)
    resnet18_backbone = ((2, 64, 2), (2, 128, 2), (2, 256, 2), (2, 512, 2))  # 18 = 1 conv + 16 conv + 1 fcn
    resnet34_backbone = ((3, 64, 2), (4, 128, 2), (6, 256, 2), (3, 512, 2))
    resnet50_backbone = ((3, 256, 3), (4, 512, 3), (6, 1024, 3), (3, 2048, 3))
    resnet101_backbone = ((3, 256, 3), (4, 512, 3), (23, 1024, 3), (3, 2048, 3))
    resnet152_backbone = ((3, 256, 3), (8, 512, 3), (36, 1024, 3), (3, 2048, 3))

    torch = dict(bias=False)
    keras = dict(bias=True)

    default_model = 'torch-resnet18'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            'torch-resnet18': dict(
                backbone_config=cls.resnet18_backbone,
                **cls.torch
            ),

            'torch-resnet34': dict(
                backbone_config=cls.resnet34_backbone,
                **cls.torch
            ),

            'torch-resnet50': dict(
                backbone_config=cls.resnet50_backbone,
                **cls.torch
            ),

            'torch-resnet101': dict(
                backbone_config=cls.resnet101_backbone,
                **cls.torch
            ),

            'torch-resnet152': dict(
                backbone_config=cls.resnet152_backbone,
                **cls.torch
            ),

            'keras-resnet50': dict(
                backbone_config=cls.resnet50_backbone,
                **cls.keras
            ),

            'keras-resnet101': dict(
                backbone_config=cls.resnet101_backbone,
                **cls.keras
            ),

            'keras-resnet152': dict(
                backbone_config=cls.resnet152_backbone,
                **cls.keras
            ),
        }


class WeightLoader(bundles.WeightLoader):
    @staticmethod
    def from_imagenet_keras(save_path):
        """see also `keras.applications.resnet`"""
        state_dict = torch_utils.Load.from_h5(save_path)
        info = []
        for k, v in state_dict.items():
            if 'kernel' in k and v.ndim <= 3:
                info.append(('w', 'l'))
            elif 'kernel' in k:
                info.append(('w', 'c'))
            elif 'bias' in k:
                info.append(('b', ''))
            elif 'gamma' in k:
                info.append(('w', 'n'))
            elif 'beta' in k:
                info.append(('b', 'n'))
            elif 'moving_mean' in k:
                info.append(('nm', 'n'))
            elif 'moving_variance' in k:
                info.append(('nv', 'n'))
            else:
                info.append(('', ''))

        key_types, value_types = math_utils.transpose(info)
        state_dict = torch_utils.Converter.tensors_from_tf_to_torch(state_dict, key_types, value_types)
        return state_dict


class WeightConverter:
    @staticmethod
    def from_torchvision(state_dict):
        """see also `torchvision.models.resnet`"""
        convert_dict = {
            'conv1': 'backbone.0.conv',
            'bn1': 'backbone.0.norm',

            'layer{0}.{1}.conv{2}.': 'backbone.{[0]+1}.{1}.fn.{[2]-1}.conv.',
            'layer{0}.{1}.bn{2}.': 'backbone.{[0]+1}.{1}.fn.{[2]-1}.norm.',
            'layer{0}.{1}.downsample.0': 'backbone.{[0]+1}.{1}.proj.conv',
            'layer{0}.{1}.downsample.1': 'backbone.{[0]+1}.{1}.proj.norm',

            'fc': 'head.linear'
        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict

    @staticmethod
    def from_keras(state_dict):
        """see also `keras.applications.resnet`
        note, not support for all versions of keras,
        as for which version of the keras it is, I don't know yet
        at least I have been successfully work on `keras==3.6.0`"""

        convert_dict = {
            'conv1_conv': 'backbone.0.conv',
            'conv1_bn': 'backbone.0.norm',

            'conv{0}_block{1}_0_conv': 'backbone.{0}.{[1]-1}.proj.conv',
            'conv{0}_block{1}_0_bn': 'backbone.{0}.{[1]-1}.proj.norm',

            'conv{0}_block{1}_{2}_conv': 'backbone.{0}.{[1]-1}.fn.{[2]-1}.conv',
            'conv{0}_block{1}_{2}_bn': 'backbone.{0}.{[1]-1}.fn.{[2]-1}.norm',

            'probs': 'head.linear'

        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


class Model(BaseImgClsModel):
    """[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
    See Also `torchvision.models.resnet`
    """

    def __init__(
            self,
            in_ch=3, input_size=224, out_features=1000, drop_prob=0.4, bias=False,
            backbone_config=Config.resnet18_backbone, add_block: nn.Module = None, block_config=dict(), **kwargs
    ):
        backbone = Backbone(backbone_config=backbone_config, bias=bias, add_block=add_block, **block_config)
        head = Linear(backbone.out_channels, out_features, mode='dl', drop_prob=drop_prob)

        super().__init__(
            in_ch=in_ch,
            input_size=input_size,
            out_features=out_features,
            backbone=backbone,
            head=head,
            **kwargs
        )


class Backbone(nn.Sequential):
    def __init__(self, in_ch=3, bias=False, backbone_config=Config.resnet18_backbone, add_block: nn.Module = None, **block_config):
        self.in_channels = in_ch
        layers = [
            Conv(in_ch, 64, 7, s=2, bias=bias),
            nn.MaxPool2d(3, stride=2, padding=1)
        ]

        in_ch = 64

        for i, (n_res, out_ch, n_conv) in enumerate(backbone_config):
            _layers = []
            for j in range(n_res):
                if i != 0 and j == 0:
                    _layers.append(ResBlock(in_ch, out_ch, n_conv, s=2, bias=bias, add_block=add_block, **block_config))
                else:
                    _layers.append(ResBlock(in_ch, out_ch, n_conv, bias=bias, add_block=add_block, **block_config))

                in_ch = out_ch
            layers.append(nn.Sequential(*_layers))

        self.out_channels = in_ch
        super().__init__(*layers)


class ResBlock(Residual):
    def __init__(self, in_ch, out_ch, n_conv=2, s=1, bias=False,
                 add_block: nn.Module = None, **block_config):
        self.in_channels = in_ch
        if n_conv == 2:
            conv_seq = nn.Sequential(
                Conv(in_ch, out_ch, k=3, s=s, bias=bias),
                Conv(out_ch, out_ch, k=3, s=1, is_act=False, bias=bias),
            )
        elif n_conv == 3:  # use bottleneck
            hidden_ch = out_ch // 4
            conv_seq = nn.Sequential(
                Conv(in_ch, hidden_ch, k=1, s=1, bias=bias),
                Conv(hidden_ch, hidden_ch, k=3, s=s, bias=bias),
                Conv(hidden_ch, out_ch, k=1, s=1, is_act=False, bias=bias),
            )

        else:
            raise ValueError(f'Not supported {n_conv = }')

        if add_block:
            conv_seq.append(add_block(out_ch, **block_config))

        conv_x = Conv(in_ch, out_ch, k=1, s=s, is_act=False, bias=bias) if in_ch != out_ch else nn.Identity()
        self.out_channels = out_ch

        super().__init__(
            fn=conv_seq,
            proj=conv_x,
            is_norm=True,
            norm=nn.ReLU(),  # take act as norm
        )
