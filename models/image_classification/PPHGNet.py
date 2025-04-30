import torch
import torch.nn.functional as F
from torch import nn

from .. import bundles
from ..layers import Conv


class Config(bundles.Config):
    tiny_backbone = dict(
        stem_ch=[48, 48, 96],
        layer_num=5,
        stage_config={
            # in_channels, mid_channels, out_channels, blocks, downsample
            "stage1": [96, 96, 224, 1, False, [2, 1]],
            "stage2": [224, 128, 448, 1, True, [1, 2]],
            "stage3": [448, 160, 512, 2, True, [2, 1]],
            "stage4": [512, 192, 768, 1, True, [2, 1]],
        }
    )

    det_small_backbone = dict(
        det=True,
        stem_ch=[64, 64, 128],
        layer_num=6,
        stage_config={
            # in_channels, mid_channels, out_channels, blocks, downsample
            "stage1": [128, 128, 256, 1, False, 2],
            "stage2": [256, 160, 512, 1, True, 2],
            "stage3": [512, 192, 768, 2, True, 2],
            "stage4": [768, 224, 1024, 1, True, 2],
        }
    )

    rec_small_backbone = dict(
        det=False,
        stem_ch=[64, 64, 128],
        layer_num=6,
        stage_config={
            # in_channels, mid_channels, out_channels, blocks, downsample
            "stage1": [128, 128, 256, 1, True, [2, 1]],
            "stage2": [256, 160, 512, 1, True, [1, 2]],
            "stage3": [512, 192, 768, 2, True, [2, 1]],
            "stage4": [768, 224, 1024, 1, True, [2, 1]],
        }
    )

    base_backbone = dict(
        stem_ch=[96, 96, 160],
        layer_num=7,
        stage_config={
            # in_channels, mid_channels, out_channels, blocks, downsample
            "stage1": [160, 192, 320, 1, False, [2, 1]],
            "stage2": [320, 224, 640, 2, True, [1, 2]],
            "stage3": [640, 256, 960, 3, True, [2, 1]],
            "stage4": [960, 288, 1280, 2, True, [2, 1]],
        }
    )

    default_model = 'det_tiny'

    @classmethod
    def make_full_config(cls):
        return {
            'det_tiny': dict(
                backbone_config=cls.tiny_backbone,
            ),
            'det_small': dict(
                backbone_config=cls.det_small_backbone,
            ),
            'det_base': dict(
                backbone_config=cls.base_backbone,
            ),

            'rec_tiny': dict(
                backbone_config=cls.tiny_backbone,
            ),
            'rec_small': dict(
                backbone_config=cls.rec_small_backbone,
            ),
            'rec_base': dict(
                backbone_config=cls.base_backbone,
            ),
        }


class WeightConverter:
    backbone_convert_dict = {
        '{0}.bn': '{0}.norm',
    }


class Backbone(nn.Module):
    def __init__(
            self,
            stem_ch,
            stage_config,
            layer_num,
            in_ch=3,
            det=False,
            out_indices=None
    ):
        super().__init__()
        self.det = det
        self.out_indices = out_indices if out_indices is not None else [0, 1, 2, 3]

        # stem
        stem_ch.insert(0, in_ch)
        self.stem = nn.Sequential(*[
            Conv(stem_ch[i], stem_ch[i + 1], 3, 2 if i == 0 else 1, bias=False, mode='cna')
            for i in range(len(stem_ch) - 1)
        ])

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if det else nn.Identity()

        # stages
        self.stages = nn.ModuleList()
        self.out_channels = []
        for block_id, k in enumerate(stage_config):
            in_ch, mid_ch, out_ch, block_num, downsample, stride = stage_config[k]
            self.stages.append(HG_Stage(in_ch, mid_ch, out_ch, block_num, layer_num, downsample, stride))
            if block_id in self.out_indices:
                self.out_channels.append(out_ch)

        if not self.det:
            self.out_channels = stage_config["stage4"][2]

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)

        out = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if self.det and i in self.out_indices:
                out.append(x)
        if self.det:
            return out

        if self.training:
            x = F.adaptive_avg_pool2d(x, [1, 40])
        else:
            x = F.avg_pool2d(x, [3, 2])
        return x


class HG_Stage(nn.Module):
    def __init__(
            self,
            in_ch,
            mid_ch,
            out_ch,
            block_num,
            layer_num,
            downsample=True,
            stride=[2, 1]
    ):
        super().__init__()
        self.downsample = Conv(in_ch, in_ch, 3, stride, 1, bias=False, groups=in_ch, mode='cn') if downsample else nn.Identity()

        blocks_list = []
        blocks_list.append(HG_Block(
            in_ch,
            mid_ch,
            out_ch,
            layer_num,
            identity=False
        ))
        for _ in range(block_num - 1):
            blocks_list.append(HG_Block(
                out_ch,
                mid_ch,
                out_ch,
                layer_num,
                identity=True
            ))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class HG_Block(nn.Module):
    def __init__(
            self,
            in_ch,
            mid_ch,
            out_ch,
            layer_num,
            identity=False,
    ):
        super().__init__()
        self.identity = identity

        self.layers = nn.ModuleList()
        self.layers.append(Conv(in_ch, mid_ch, 3, 1, bias=False, mode='cna'))
        for _ in range(layer_num - 1):
            self.layers.append(Conv(mid_ch, mid_ch, 3, 1, bias=False, mode='cna'))

        # feature aggregation
        total_ch = in_ch + layer_num * mid_ch
        self.aggregation_conv = Conv(total_ch, out_ch, 1, 1, bias=False, mode='cna')
        self.att = ESEModule(out_ch)

    def forward(self, x):
        identity = x
        output = [x]
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.aggregation_conv(x)
        x = self.att(x)
        if self.identity:
            x += identity
        return x


class ESEModule(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(
            in_channels=ch,
            out_channels=ch,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x * identity
