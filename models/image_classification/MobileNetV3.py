from functools import partial

from torch import nn

from utils import torch_utils, math_utils
from . import BaseImgClsModel
from .. import bundles
from ..layers import Conv, Linear


class Config(bundles.Config):
    large_backbone = (
        # in_ch, k, hidden_ch, out_ch, use_se, act_type, stride, dilation
        (16, 3, 16, 16, False, "RE", 1, 1),
        (16, 3, 64, 24, False, "RE", 2, 1),  # C1
        (24, 3, 72, 24, False, "RE", 1, 1),
        (24, 5, 72, 40, True, "RE", 2, 1),  # C2
        (40, 5, 120, 40, True, "RE", 1, 1),
        (40, 5, 120, 40, True, "RE", 1, 1),
        (40, 3, 240, 80, False, "HS", 2, 1),  # C3
        (80, 3, 200, 80, False, "HS", 1, 1),
        (80, 3, 184, 80, False, "HS", 1, 1),
        (80, 3, 184, 80, False, "HS", 1, 1),
        (80, 3, 480, 112, True, "HS", 1, 1),
        (112, 3, 672, 112, True, "HS", 1, 1),
        (112, 5, 672, 160, True, "HS", 2, 1),  # C4
        (160, 5, 960, 160, True, "HS", 1, 1),
        (160, 5, 960, 160, True, "HS", 1, 1),
    )
    small_backbone = (
        (16, 3, 16, 16, True, "RE", 2, 1),  # C1
        (16, 3, 72, 24, False, "RE", 2, 1),  # C2
        (24, 3, 88, 24, False, "RE", 1, 1),
        (24, 5, 96, 40, True, "HS", 2, 1),  # C3
        (40, 5, 240, 40, True, "HS", 1, 1),
        (40, 5, 240, 40, True, "HS", 1, 1),
        (40, 5, 120, 48, True, "HS", 1, 1),
        (48, 5, 144, 48, True, "HS", 1, 1),
        (48, 5, 288, 96, True, "HS", 2, 1),  # C4
        (96, 5, 576, 96, True, "HS", 1, 1),
        (96, 5, 576, 96, True, "HS", 1, 1),
    )

    large_head = dict(
        hidden_ch=1280,
        drop_prob=0.2
    )

    small_head = dict(
        hidden_ch=1024,
        drop_prob=0.2
    )

    default_model = 'large'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            'large': dict(
                backbone_config=cls.large_backbone,
                head_config=cls.large_head
            ),
            'small': dict(
                backbone_config=cls.small_backbone,
                head_config=cls.small_head
            )
        }


class WeightConverter:
    @staticmethod
    def from_torchvision(state_dict):
        convert_dict = {
            'features': 'backbone',
            'features.{0}.block.2.fc1': 'backbone.{0}.block.2.scale.1.0',
            'features.{0}.block.2.fc2': 'backbone.{0}.block.2.scale.2.0',
            'classifier.0': 'head.0.linear',
            'classifier.3': 'head.1'
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


def make_norm_fn():
    return partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)


class Model(BaseImgClsModel):
    """[Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf)
    See Also `torchvision.models.mobilenet`
    """

    def __init__(
            self,
            in_ch, input_size, out_features,
            backbone_config=Config.large_backbone, head_config=Config.large_head, **kwargs
    ):
        backbone = Backbone(in_ch, backbone_config=backbone_config)
        head = nn.Sequential(
            Linear(backbone.out_channels, head_config['hidden_ch'], mode='lad', act=nn.Hardswish(inplace=True), drop_prob=head_config['drop_prob']),
            nn.Linear(head_config['hidden_ch'], out_features),
        )

        super().__init__(
            in_ch=in_ch,
            input_size=input_size,
            out_features=out_features,
            backbone=backbone,
            head=head,
            **kwargs
        )


class Backbone(nn.Sequential):
    def __init__(self, in_ch, backbone_config=Config.large_backbone):
        out_ch = backbone_config[0][0]
        layers = [
            Conv(in_ch, out_ch, 3, 2, bias=False, mode='cna', norm=nn.BatchNorm2d(16), act=nn.Hardswish(), detail_name=False),
        ]

        for cfg in backbone_config:
            layers.append(InvertedResidual(*cfg))

        in_ch = backbone_config[-1][3]
        out_ch = 6 * in_ch
        layers.append(Conv(in_ch, out_ch, 1, bias=False, mode='cna', norm_fn=make_norm_fn(), act=nn.Hardswish(), detail_name=False))

        super().__init__(*layers)
        self.out_channels = out_ch


class InvertedResidual(nn.Module):
    def __init__(self, in_ch, k, hidden_ch, out_ch, use_se, act_type, stride, dilation):
        super().__init__()
        self.use_res_connect = stride == 1 and in_ch == out_ch

        layers = []
        act = nn.Hardswish() if act_type == 'HS' else nn.ReLU()

        # expand
        if hidden_ch != in_ch:
            layers.append(Conv(in_ch, hidden_ch, 1, bias=False, mode='cna', norm_fn=make_norm_fn(), act=act, detail_name=False))

        # depthwise
        stride = 1 if dilation > 1 else stride
        layers.append(Conv(hidden_ch, hidden_ch, k, stride, dilation=dilation, groups=hidden_ch, bias=False, mode='cna', norm_fn=make_norm_fn(), act=act, detail_name=False))
        if use_se:
            squeeze_ch = math_utils.make_divisible(hidden_ch // 4, 8)
            layers.append(SqueezeExcitation(hidden_ch, squeeze_ch))

        layers.append(Conv(hidden_ch, out_ch, 1, bias=False, mode='cn', norm_fn=make_norm_fn(), detail_name=False))

        self.block = nn.Sequential(*layers)
        self.out_channels = out_ch

    def forward(self, x):
        y = self.block(x)
        if self.use_res_connect:
            y += x
        return y


class SqueezeExcitation(nn.Module):
    def __init__(self, in_ch, hidden_ch, act_fn=nn.ReLU, scale_act_fn=nn.Hardsigmoid):
        super().__init__()
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(in_ch, hidden_ch, 1, mode='ca', act=act_fn(), detail_name=False),
            Conv(hidden_ch, in_ch, 1, mode='ca', act=scale_act_fn(), detail_name=False)
        )

    def forward(self, x):
        scale = self.scale(x)
        return scale * x
