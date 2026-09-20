import math

import torch
import torch.nn.functional as F
from torch import nn

from .. import bundles
from ..layers import Conv

NET_CONFIG_DET = {
    "tiny": {
        # stem(mid=16, out=32)  channels: 32 → 48 → 64 → 160
        "stem": (16, 32),
        "blocks_s1": [[3, 32, 32, 1, True], [3, 32, 32, 1, False]],
        "blocks_s2": [
            [3, 32, 48, 2, False],
            [3, 48, 48, 1, True],
            [3, 48, 48, 1, False],
        ],
        "blocks_s3": [
            [3, 48, 64, 2, False],
            [3, 64, 64, 1, True],
            [3, 64, 64, 1, False],
            [3, 64, 64, 1, True],
            [3, 64, 64, 1, False],
        ],
        "blocks_s4": [
            [3, 64, 160, 2, False],
            [3, 160, 160, 1, True],
            [3, 160, 160, 1, False],
        ],
    },
    "small": {
        # stem(mid=24, out=48)  channels: 48 → 96 → 192 → 384
        "stem": (24, 48),
        "blocks_s1": [[3, 48, 48, 1, True], [3, 48, 48, 1, False]],
        "blocks_s2": [
            [3, 48, 96, 2, False],
            [3, 96, 96, 1, True],
            [3, 96, 96, 1, False],
        ],
        "blocks_s3": [
            [3, 96, 192, 2, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
        ],
        "blocks_s4": [
            [3, 192, 384, 2, False],
            [3, 384, 384, 1, True],
            [3, 384, 384, 1, False],
        ],
    },
    "medium": {
        # stem(mid=64, out=128)  channels: 128 → 256 → 512 → 896
        "stem": (64, 128),
        "blocks_s1": [[3, 128, 128, 1, True], [3, 128, 128, 1, False]],
        "blocks_s2": [
            [3, 128, 256, 2, False],
            [3, 256, 256, 1, True],
            [3, 256, 256, 1, False],
        ],
        "blocks_s3": [
            [3, 256, 512, 2, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
        ],
        "blocks_s4": [
            [3, 512, 896, 2, False],
            [3, 896, 896, 1, True],
            [3, 896, 896, 1, False],
        ],
    },
}

NET_CONFIG_REC = {
    "tiny": {
        # stem: simple (2×Conv2D_BN+GELU, mid=24, out=48)  channels: 48 → 96 → 160
        "stem": (24, 48),
        "stem_type": "simple",
        "blocks2": [[3, 48, 48, 1, True]],
        "blocks3": [[3, 48, 48, 1, False]],
        "blocks4": [
            [3, 48, 96, (2, 1), False],
            [3, 96, 96, 1, True],
            [3, 96, 96, 1, False],
        ],
        "blocks5": [
            [3, 96, 160, (2, 1), False],
            [3, 160, 160, 1, True],
            [3, 160, 160, 1, False],
            [3, 160, 160, 1, False],
        ],
        "blocks6": [],
    },
    "small": {
        # stem: branch StemBlock (mid=48, out=96)  channels: 96 → 192 → 384
        "stem": (48, 96),
        "stem_type": "branch",
        "blocks2": [[3, 96, 96, 1, True]],
        "blocks3": [[3, 96, 96, 1, False], [3, 96, 96, 1, False]],
        "blocks4": [
            [3, 96, 192, (2, 1), False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
            [3, 192, 192, 1, True],
            [3, 192, 192, 1, False],
        ],
        "blocks5": [
            [3, 192, 384, (2, 1), False],
            [3, 384, 384, 1, True],
            [3, 384, 384, 1, False],
        ],
        "blocks6": [],
    },
    "medium": {
        # stem: branch StemBlock (mid=64, out=128)  channels: 128 → 256 → 512 → 768
        "stem": (64, 128),
        "stem_type": "branch",
        "blocks2": [[3, 128, 128, 1, True]],
        "blocks3": [
            [3, 128, 256, 1, False],
            [3, 256, 256, 1, False],
            [3, 256, 256, 1, True],
        ],
        "blocks4": [
            [3, 256, 512, (2, 1), False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
            [3, 512, 512, 1, True],
            [3, 512, 512, 1, False],
        ],
        "blocks5": [
            [3, 512, 768, (2, 1), False],
            [3, 768, 768, 1, True],
            [3, 768, 768, 1, False],
        ],
        "blocks6": [],
    },
}


class Config(bundles.Config):
    det_tiny_backbone = dict(
        det=True,
        model_size='tiny'
    )
    det_small_backbone = dict(
        det=True,
        model_size='small'
    )
    det_medium_backbone = dict(
        det=True,
        model_size='medium'
    )

    rec_tiny_backbone = dict(
        det=False,
        model_size='tiny'
    )
    rec_small_backbone = dict(
        det=False,
        model_size='small'
    )
    rec_medium_backbone = dict(
        det=False,
        model_size='medium'
    )

    default_model = 'det_small'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            'det_tiny': dict(
                backbone_config=cls.det_tiny_backbone,
            ),
            'det_small': dict(
                backbone_config=cls.det_small_backbone,
            ),
            'det_medium': dict(
                backbone_config=cls.det_medium_backbone,
            ),
            'rec_tiny': dict(
                backbone_config=cls.rec_tiny_backbone,
            ),
            'rec_small': dict(
                backbone_config=cls.rec_small_backbone,
            ),
            'rec_medium': dict(
                backbone_config=cls.rec_medium_backbone,
            ),
        }


class WeightConverter:
    backbone_convert_dict = {
        'backbone.{0}.bn': 'backbone.{0}.norm',
    }

    transformers_backbone_convert_dict = {
        'model.backbone.encoder.blocks.{0}.blocks.{1}.channel_conv1.convolution': 'backbone.blocks{[0]+2}.{1}.channel_mixer.expand.conv',
        'model.backbone.encoder.blocks.{0}.blocks.{1}.channel_conv1.normalization': 'backbone.blocks{[0]+2}.{1}.channel_mixer.expand.norm',
        'model.backbone.encoder.blocks.{0}.blocks.{1}.channel_conv2.convolution': 'backbone.blocks{[0]+2}.{1}.channel_mixer.compress.conv',
        'model.backbone.encoder.blocks.{0}.blocks.{1}.channel_conv2.normalization': 'backbone.blocks{[0]+2}.{1}.channel_mixer.compress.norm',
        # 'model.backbone.encoder.blocks.0.blocks.0.token_conv.bias': 'backbone.blocks2.0.token_mixer.rep_dw.conv.conv.weight',
        'model.backbone.encoder.blocks.{0}.blocks.{1}.token_squeeze_excitation.convolutions.0': 'backbone.blocks{[0]+2}.{1}.token_mixer.se.conv1',
        'model.backbone.encoder.blocks.{0}.blocks.{1}.token_squeeze_excitation.convolutions.2': 'backbone.blocks{[0]+2}.{1}.token_mixer.se.conv2',

        'model.backbone.encoder.convolution.{0}.convolution': 'backbone.conv1.{0}.conv',
        'model.backbone.encoder.convolution.{0}.normalization': 'backbone.conv1.{0}.norm',

        # todo: is that different from transformers' weight and paddle weight?
        'model.backbone.encoder.blocks.{0}.blocks.{1}.token_conv.convolution': 'backbone.blocks{[0]+2}.{1}.token_mixer.rep_dw.conv.conv',
        'model.backbone.encoder.blocks.{0}.blocks.{1}.token_conv.normalization': 'backbone.blocks{[0]+2}.{1}.token_mixer.rep_dw.conv.norm',
        'model.backbone.encoder.blocks.{0}.blocks.{1}.token_conv': 'backbone.blocks{[0]+2}.{1}.token_mixer.dw_conv.conv',
    }


class Backbone(nn.Module):
    def __init__(
            self,
            det=False,
            model_size='small',
            in_ch=3,
            **kwargs
    ):
        super().__init__()
        self.det = det

        if det:
            assert model_size in NET_CONFIG_DET, f"det model_size must be one of {list(NET_CONFIG_DET.keys())} but got '{model_size}'"
            cfg = NET_CONFIG_DET[model_size]
            stem_mid, stem_out = cfg["stem"]
            self.stem = StemBlock(in_ch, stem_mid, stem_out)

            def make_det_stage(key):
                return nn.Sequential(*[
                    LCNetV4Block(in_c, out_c, s, k, se)
                    for k, in_c, out_c, s, se in cfg[key]
                ])

            self.blocks_s1 = make_det_stage("blocks_s1")
            self.blocks_s2 = make_det_stage("blocks_s2")
            self.blocks_s3 = make_det_stage("blocks_s3")
            self.blocks_s4 = make_det_stage("blocks_s4")
            self.out_channels = [
                cfg["blocks_s1"][-1][2],
                cfg["blocks_s2"][-1][2],
                cfg["blocks_s3"][-1][2],
                cfg["blocks_s4"][-1][2],
            ]
        else:
            assert model_size in NET_CONFIG_REC, f"rec model_size must be one of {list(NET_CONFIG_REC.keys())} but got '{model_size}'"
            cfg = NET_CONFIG_REC[model_size]
            stem_mid, stem_out = cfg["stem"]
            if cfg["stem_type"] == "branch":
                self.conv1 = StemBlock(in_ch, stem_mid, stem_out)
            else:
                self.conv1 = nn.Sequential(
                    Conv(in_ch, stem_mid, 3, 2, bias=False, mode='cn'),
                    nn.GELU(),
                    Conv(stem_mid, stem_out, 3, 2, bias=False, mode='cn'),
                )

            def make_rec_stage(stage_name):
                return nn.Sequential(*[
                    LCNetV4Block(in_c, out_c, s, k, se)
                    for k, in_c, out_c, s, se in cfg.get(stage_name, [])
                ])

            self.blocks2 = make_rec_stage("blocks2")
            self.blocks3 = make_rec_stage("blocks3")
            self.blocks4 = make_rec_stage("blocks4")
            self.blocks5 = make_rec_stage("blocks5")
            self.blocks6 = make_rec_stage("blocks6")

            self.out_channels = stem_out
            for sname in reversed(["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]):
                if cfg.get(sname):
                    self.out_channels = cfg[sname][-1][2]
                    break

    def forward(self, x):
        if self.det:
            x = self.stem(x)
            o1 = self.blocks_s1(x)
            o2 = self.blocks_s2(o1)
            o3 = self.blocks_s3(o2)
            o4 = self.blocks_s4(o3)
            return [o1, o2, o3, o4]

        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)
        if self.training:
            x = F.adaptive_avg_pool2d(x, [1, 40])
        else:
            x = F.avg_pool2d(x, [3, 2])
        return x


class StemBlock(nn.Module):
    def __init__(self, in_ch=3, mid_ch=48, out_ch=96):
        super().__init__()
        self.stem1 = Conv(in_ch, mid_ch, 3, 2, bias=False, mode='cna')
        self.stem2a = SamePadConv(mid_ch, mid_ch // 2, 2, 1, bias=False, mode='cna')
        self.stem2b = SamePadConv(mid_ch // 2, mid_ch, 2, 1, bias=False, mode='cna')
        self.stem3 = Conv(mid_ch * 2, mid_ch, 3, 2, bias=False, mode='cna')
        self.stem4 = Conv(mid_ch, out_ch, 1, 1, bias=False, mode='cna')
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)

    def forward(self, x):
        x = self.stem1(x)
        x2 = self.stem2b(self.stem2a(x))
        x1 = self.pool(pad_same(x, 2, 1))
        x = self.stem4(self.stem3(torch.cat([x1, x2], dim=1)))
        return x


def pad_same(x, kernel_size, stride=1):
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size
    if isinstance(stride, int):
        sh = sw = stride
    else:
        sh, sw = stride

    _, _, h, w = x.shape

    def get_pad(in_size, k, s):
        out_size = math.ceil(in_size / s)
        pad = max((out_size - 1) * s + k - in_size, 0)
        return pad // 2, pad - pad // 2

    pad_top, pad_bottom = get_pad(h, kh, sh)
    pad_left, pad_right = get_pad(w, kw, sw)
    if pad_top or pad_bottom or pad_left or pad_right:
        x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])
    return x


class SamePadConv(Conv):
    def __init__(self, in_ch, out_ch, k, s=1, **kwargs):
        super().__init__(in_ch, out_ch, k, s, p=0, **kwargs)

    def forward(self, input):
        return super().forward(pad_same(input, self.kernel_size, self.stride))


class SELayer(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch // r, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(ch // r, ch, 1)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        x = x.mean(dim=[2, 3], keepdim=True)
        x = self.relu(self.conv1(x))
        x = self.hardsigmoid(self.conv2(x))
        return identity * x


class RepDWConv(nn.Module):
    def __init__(self, ch, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = Conv(ch, ch, kernel_size, 1, padding, groups=ch, bias=False, mode='cn')
        self.conv1 = nn.Conv2d(ch, ch, 1, 1, 0, groups=ch, bias=False)
        self.norm = nn.BatchNorm2d(ch)

    def forward(self, x):
        return self.norm(self.conv(x) + self.conv1(x) + x)


class LCNetV4Block(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            stride,
            dw_size,
            use_se=False,
            expand_ratio=2,
    ):
        super().__init__()
        self.has_residual = in_ch == out_ch and stride == 1
        self.use_rep_dw = stride == 1 and in_ch == out_ch

        self.token_mixer = nn.Sequential()
        if self.use_rep_dw:
            self.token_mixer.add_module('rep_dw', RepDWConv(in_ch, dw_size))
        else:
            padding = (dw_size - 1) // 2
            self.token_mixer.add_module(
                'dw_conv',
                Conv(in_ch, in_ch, dw_size, stride, padding, groups=in_ch, bias=False, mode='cn')
            )
        if use_se:
            self.token_mixer.add_module('se', SELayer(in_ch))

        hidden_ch = int(in_ch * expand_ratio)
        self.channel_mixer = nn.Sequential()
        self.channel_mixer.add_module('expand', Conv(in_ch, hidden_ch, 1, 1, 0, bias=False, mode='cn'))
        self.channel_mixer.add_module('act', nn.GELU())
        self.channel_mixer.add_module('compress', Conv(hidden_ch, out_ch, 1, 1, 0, bias=False, mode='cn'))

    def forward(self, x):
        x = self.token_mixer(x)
        if self.has_residual:
            return x + self.channel_mixer(x)
        return self.channel_mixer(x)
