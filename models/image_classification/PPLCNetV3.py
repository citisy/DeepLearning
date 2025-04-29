from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from utils import math_utils
from .. import bundles
from ..layers import Conv


class Config(bundles.Config):
    det_backbone = dict(
        det=True,
        block_config={
            # k, in_c, out_c, s, use_se
            "blocks2": [[3, 16, 32, 1, False]],
            "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
            "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
            "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                        [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
            "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True],
                        [5, 512, 512, 1, False], [5, 512, 512, 1, False]]
        }
    )

    rec_backbone = dict(
        det=False,
        scale=0.95,
        block_config={
            # k, in_c, out_c, s, use_se
            "blocks2": [[3, 16, 32, 1, False]],
            "blocks3": [[3, 32, 64, 1, False], [3, 64, 64, 1, False]],
            "blocks4": [[3, 64, 128, (2, 1), False], [3, 128, 128, 1, False]],
            "blocks5": [[3, 128, 256, (1, 2), False], [5, 256, 256, 1, False],
                        [5, 256, 256, 1, False], [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
            "blocks6": [[5, 256, 512, (2, 1), True], [5, 512, 512, 1, True],
                        [5, 512, 512, (2, 1), False], [5, 512, 512, 1, False]]
        }
    )

    default_model = 'det'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            'det': dict(
                backbone_config=cls.det_backbone,
            ),

            'rec': dict(
                backbone_config=cls.rec_backbone,
            )
        }


class Backbone(nn.Module):
    def __init__(
            self,
            scale=1.0,
            conv_kxk_num=4,
            det=True,
            block_config={},
            **kwargs
    ):
        super().__init__()
        self.scale = scale
        self.det = det

        make_divisible = partial(math_utils.make_divisible, divisor=16)

        self.conv1 = Conv(3, make_divisible(16 * scale), 3, 2, (3 - 1) // 2, bias=False, mode='cn', norm_fn=nn.BatchNorm2d)

        for name, config in block_config.items():
            layers = []
            for i, (k, in_c, out_c, s, se) in enumerate(config):
                block = LCNetV3Block(
                    in_ch=make_divisible(in_c * scale),
                    out_ch=make_divisible(out_c * scale),
                    dw_size=k,
                    stride=s,
                    use_se=se,
                    conv_kxk_num=conv_kxk_num,
                )
                layers.append(block)
            self.register_module(name, nn.Sequential(*layers))

        self.out_channels = make_divisible(512 * scale)

        if self.det:
            mv_c = [16, 24, 56, 480]
            in_ches = [make_divisible(block_config[name][-1][2] * scale) for name in ['blocks3', 'blocks4', 'blocks5', 'blocks6']]
            out_ches = [int(mv * scale) for mv in mv_c]

            layers = []
            for in_ch, out_ch in zip(in_ches, out_ches):
                layers.append(
                    nn.Conv2d(in_ch, out_ch, 1, 1, 0)
                )

            self.layer_list = nn.ModuleList(layers)
            self.out_channels = out_ches

    def forward(self, x):
        out_list = []
        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        out_list.append(x)

        x = self.blocks4(x)
        out_list.append(x)

        x = self.blocks5(x)
        out_list.append(x)

        x = self.blocks6(x)
        out_list.append(x)

        if self.det:
            out_list[0] = self.layer_list[0](out_list[0])
            out_list[1] = self.layer_list[1](out_list[1])
            out_list[2] = self.layer_list[2](out_list[2])
            out_list[3] = self.layer_list[3](out_list[3])
            return out_list

        if self.training:
            x = F.adaptive_avg_pool2d(x, [1, 40])
        else:
            x = F.avg_pool2d(x, [3, 2])
        return x


class LCNetV3Block(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            stride,
            dw_size,
            use_se=False,
            conv_kxk_num=4,
    ):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = LearnableRepBlock(
            in_ch=in_ch,
            out_ch=in_ch,
            kernel_size=dw_size,
            stride=stride,
            groups=in_ch,
            num_conv_branches=conv_kxk_num,
        )
        if use_se:
            self.se = SEBlock(in_ch)

        self.pw_conv = LearnableRepBlock(
            in_ch=in_ch,
            out_ch=out_ch,
            kernel_size=1,
            stride=1,
            num_conv_branches=conv_kxk_num,
        )

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(
            Conv(ch, ch // r, 1, act=nn.ReLU(), mode='ca'),
            Conv(ch // r, ch, 1, act=nn.Hardsigmoid(), mode='ca')
        )

    def forward(self, x):
        y = self.sq(x)
        y = self.ex(y)
        x = x * y

        return x


class LearnableRepBlock(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size,
            stride=1,
            groups=1,
            num_conv_branches=1,
    ):
        super().__init__()
        self.identity = nn.BatchNorm2d(
            num_features=in_ch,
        ) if out_ch == in_ch and stride == 1 else None

        self.conv_kxk = nn.ModuleList([
            Conv(in_ch, out_ch, kernel_size, stride, (kernel_size - 1) // 2, bias=False, groups=groups, mode='cn', norm_fn=nn.BatchNorm2d)
            for _ in range(num_conv_branches)
        ])

        self.conv_1x1 = Conv(in_ch, out_ch, 1, stride, 0, bias=False, groups=groups, mode='cn', norm_fn=nn.BatchNorm2d) if kernel_size > 1 else None

        self.lab = LearnableAffineBlock()
        self.act = nn.Sequential(
            nn.Hardswish(inplace=True),
            LearnableAffineBlock()
        )

        self.stride = stride

    def forward(self, x):
        out = 0
        if self.identity is not None:
            out += self.identity(x)

        if self.conv_1x1 is not None:
            out += self.conv_1x1(x)

        for conv in self.conv_kxk:
            out += conv(x)

        out = self.lab(out)
        if self.stride != 2:
            out = self.act(out)
        return out


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor([scale_value]))
        self.bias = nn.Parameter(torch.Tensor([bias_value]))

    def forward(self, x):
        return self.scale * x + self.bias
