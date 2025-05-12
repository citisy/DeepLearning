import torch
import torch.nn.functional as F
from torch import nn

from utils import torch_utils
from . import Unet
from .. import bundles
from ..layers import Conv, Residual, Cache


class Config(bundles.Config):
    # (mid_ch, n_layer)
    base_backbone_down_layer_configs = [
        (32, 7),
        (32, 6),
        (64, 5),
        (128, 4),
        (256, 4),
    ]

    base_backbone_mid_layer_config = (256, 4)

    base_backbone_up_layer_configs = [
        (16, 7),
        (32, 6),
        (64, 5),
        (128, 4),
        (256, 4),
    ]

    base_backbone_configs = dict(
        out_ch=64,
        unit_ch=64,
        ch_mult=(1, 2, 4, 8, 8, 8)
    )

    base_head_in_ches = (64, 64, 128, 256, 512, 512)

    small_backbone_down_layer_configs = [
        (16, 7),
        (16, 6),
        (16, 5),
        (16, 4),
        (16, 4),
    ]

    small_backbone_mid_layer_config = (16, 4)

    small_backbone_up_layer_configs = [
        (16, 7),
        (16, 6),
        (16, 5),
        (16, 4),
        (16, 4),
    ]

    small_backbone_configs = dict(
        out_ch=64,
        unit_ch=64,
        ch_mult=(1, 1, 1, 1, 1, 1)
    )

    small_head_in_ches = (64, 64, 64, 64, 64, 64)

    default_model = 'base'

    @classmethod
    def make_full_config(cls) -> dict:
        """see https://github.com/xuebinqin/U-2-Net?tab=readme-ov-file"""

        return {
            'base': dict(
                backbone_config=dict(
                    down_layer_configs=cls.base_backbone_down_layer_configs,
                    mid_layer_config=cls.base_backbone_mid_layer_config,
                    up_layer_configs=cls.base_backbone_up_layer_configs,
                    **cls.base_backbone_configs
                ),
                head_config=dict(
                    in_ches=cls.base_head_in_ches
                )
            ),

            'small': dict(
                backbone_config=dict(
                    down_layer_configs=cls.small_backbone_down_layer_configs,
                    mid_layer_config=cls.small_backbone_mid_layer_config,
                    up_layer_configs=cls.small_backbone_up_layer_configs,
                    **cls.small_backbone_configs
                ),
                head_config=dict(
                    in_ches=cls.small_head_in_ches
                )
            )
        }


class WeightConverter:
    @classmethod
    def from_official(cls, state_dict):
        s = '"conv" if "[1]" == "conv" else "norm"'
        convert_dict = {
            'stage{2}.rebnconvin.{1}_s1.': 'backbone.downs.{[2]-1}.1.0.{%s}.' % s,
            'stage{2}.rebnconv{0}.{1}_s1.': 'backbone.downs.{[2]-1}.1.1.fn.downs.{[0]-1}.1.{%s}.' % s,

            'stage5.rebnconv4.{1}_s1.': 'backbone.downs.4.1.1.fn.mid.{%s}.' % s,
            'stage5.rebnconv{0}d.{1}_s1.': 'backbone.downs.4.1.1.fn.ups.{3-[0]}.0.{%s}.' % s,

            'stage6.rebnconvin.{1}_s1.': 'backbone.mid.1.0.{%s}.' % s,
            'stage6.rebnconv{0}.{1}_s1.': 'backbone.mid.1.1.fn.downs.{[0]-1}.1.{%s}.' % s,
            'stage6.rebnconv4.{1}_s1.': 'backbone.mid.1.1.fn.mid.{%s}.' % s,
            'stage6.rebnconv{0}d.{1}_s1.': 'backbone.mid.1.1.fn.ups.{3-[0]}.0.{%s}.' % s,

            'stage{2}d.rebnconvin.{1}_s1.': 'backbone.ups.{5-[2]}.0.0.{%s}.' % s,
            'stage{2}d.rebnconv{0}.{1}_s1.': 'backbone.ups.{5-[2]}.0.1.fn.downs.{[0]-1}.1.{%s}.' % s,

            'stage5d.rebnconv4.{1}_s1.': 'backbone.ups.0.0.1.fn.mid.{%s}.' % s,
            'stage5d.rebnconv{0}d.{1}_s1.': 'backbone.ups.0.0.1.fn.ups.{3-[0]}.0.{%s}.' % s,

            'side{0}.': 'head.sides.{[0]-1}.',
            'outconv': 'head.to_out.conv'
        }

        # down layers
        for i in range(1, 5):
            convert_dict[f'stage{i}.rebnconv{8 - i}.{{1}}_s1.'] = f'backbone.downs.{i - 1}.1.1.fn.mid.{{%s}}.' % s
            convert_dict[f'stage{i}.rebnconv{{0}}d.{{1}}_s1.'] = f'backbone.downs.{i - 1}.1.1.fn.ups.{{{7 - i}-[0]}}.0.{{%s}}.' % s

        # up layers
        for i in range(1, 5):
            convert_dict[f'stage{i}d.rebnconv{8 - i}.{{1}}_s1.'] = f'backbone.ups.{5 - i}.0.1.fn.mid.{{%s}}.' % s
            convert_dict[f'stage{i}d.rebnconv{{0}}d.{{1}}_s1.'] = f'backbone.ups.{5 - i}.0.1.fn.ups.{{{7 - i}-[0]}}.0.{{%s}}.' % s

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


class Model(nn.Module):
    """[U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/pdf/2005.09007)"""

    def __init__(self, in_ch=3, out_ch=1, backbone_config={}, head_config={}, **kwargs):
        super().__init__()
        self.backbone = TopUnet(in_ch=in_ch, **backbone_config)
        hook_layers = torch_utils.ModuleManager.get_module_by_key(self.backbone, Cache, is_return_last_module=True)
        hook_features = [layer[0].features for layer in hook_layers[::-1]]
        self.head = Head(out_ch, hook_features, **head_config)
        self.criterion = nn.BCELoss(size_average=True)

    def forward(self, x, label_masks=None):
        x = self.backbone(x)
        preds, hidden_states = self.head(x)

        if self.training:
            return self.loss(preds, hidden_states, label_masks)
        else:
            return dict(
                preds=preds,
                hidden_states=hidden_states  # note, haven't sigmoid
            )

    def loss(self, preds, hidden_states, label_masks=None):
        losses = [self.criterion(preds, label_masks)]
        for hidden_state in hidden_states:
            hidden_state = F.sigmoid(hidden_state)
            losses.append(self.criterion(hidden_state, label_masks))

        loss = torch.cat(losses).sum()
        return dict(
            loss=loss,
            losses=losses,
        )


class TopUnet(Unet.CirUnetBlock):
    down_layer_configs = Config.base_backbone_down_layer_configs
    mid_layer_config = Config.base_backbone_mid_layer_config
    up_layer_configs = Config.base_backbone_up_layer_configs

    def make_down_layer(self, in_ch, out_ch, is_top=False, is_bottom=False, layer_idx=None):
        if is_top:
            layer = nn.Sequential(
                nn.Identity(),  # for layer alignment
                make_RSU_block(in_ch, out_ch, *self.down_layer_configs[layer_idx])
            )
        else:
            layer = nn.Sequential(
                nn.MaxPool2d(2, stride=2, ceil_mode=True),
                make_RSU_block(in_ch, out_ch, *self.down_layer_configs[layer_idx], flag=is_bottom),
            )

        return layer

    def make_mid_layer(self, in_ch, out_ch, layer_idx=None):
        return nn.Sequential(
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            make_RSU_block(in_ch, out_ch, *self.mid_layer_config, flag=True),
            Cache(0, inplace=True, init_features=[None]),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

    def make_up_layer(self, in_ch, out_ch, is_top=False, is_bottom=False, layer_idx=None):
        if is_top:
            layer = nn.Sequential(
                make_RSU_block(in_ch * 2, out_ch, *self.up_layer_configs[layer_idx]),
            )
        else:
            layer = nn.Sequential(
                make_RSU_block(in_ch * 2, out_ch, *self.up_layer_configs[layer_idx], flag=is_bottom),
                Cache(0, inplace=True, init_features=[None]),
                nn.Upsample(scale_factor=2, mode='bilinear')
            )

        return layer

    def forward(self, x):
        hs = []
        for down in self.downs:
            x = down(x)
            hs.append(x)

        x = self.mid(x)

        for up in self.ups:
            h = hs.pop()
            x = torch.cat([x, h], 1)
            x = up(x)

        return x


def make_RSU_block(in_ch, out_ch, unit_ch, n_layer, flag=False):
    """ReSidual U-block(RSU)"""
    return nn.Sequential(
        Conv(in_ch, out_ch, 3, mode='cna'),
        Residual(
            BottomUnet(out_ch, out_ch, unit_ch, ch_mult=[1] * n_layer, flag=flag),
            is_norm=False
        )
    )


class BottomUnet(Unet.CirUnetBlock):
    # False -> RSU-L
    # True -> RSU-LF, dilated version, here, I even don't know the meaning of 'F' yet :-)
    flag = False

    def make_down_layer(self, in_ch, out_ch, is_top=False, is_bottom=False, layer_idx=None):
        factor = 2 ** layer_idx if self.flag else 1
        if is_top:
            layer = nn.Sequential(
                nn.Identity(),  # for layer alignment
                Conv(in_ch, out_ch, 3, mode='cna', p=factor, dilation=factor)
            )
        else:
            layer = nn.Sequential(
                nn.Identity() if self.flag else nn.MaxPool2d(2, stride=2, ceil_mode=True),
                Conv(in_ch, out_ch, 3, mode='cna', p=factor, dilation=factor)
            )

        return layer

    def make_mid_layer(self, in_ch, out_ch, layer_idx=None):
        factor = 2 ** layer_idx if self.flag else 2
        return Conv(in_ch, out_ch, 3, mode='cna', p=factor, dilation=factor)

    def make_up_layer(self, in_ch, out_ch, is_top=False, is_bottom=False, layer_idx=None):
        factor = 2 ** layer_idx if self.flag else 1
        if is_top:
            layer = nn.Sequential(
                Conv(in_ch * 2, out_ch, 3, mode='cna', p=factor, dilation=factor)
            )
        else:
            layer = nn.Sequential(
                Conv(in_ch * 2, out_ch, 3, mode='cna', p=factor, dilation=factor),
                nn.Identity() if self.flag else nn.Upsample(scale_factor=2, mode='bilinear')
            )

        return layer


class Head(nn.Module):
    def __init__(self, out_ch, hook_features, in_ches=Config.base_head_in_ches):
        super().__init__()

        layers = []
        for in_ch in in_ches:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))

        self.sides = nn.ModuleList(layers)
        self.to_out = Conv(6 * out_ch, out_ch, 1, mode='ca', act=nn.Sigmoid())
        self.hook_features = hook_features

    def forward(self, x):
        y0 = self.sides[0](x)
        hidden_states = [y0]
        for feature, layer in zip(self.hook_features, self.sides[1:]):
            y = layer(feature[0])
            y = F.upsample(y, size=y0.shape[2:], mode='bilinear')
            hidden_states.append(y)

        out = self.to_out(torch.cat(hidden_states, dim=1))
        return out, hidden_states
