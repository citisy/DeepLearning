from torch import nn
from .. import Conv, Linear, ConvInModule, OutModule
from .ShuffleNetV1 import shuffle

# (l, m, b)
L4M8_config = (4, 8, 6)
L24M2_config = (24, 2, 6)
L32M26_config = (32, 26, 6)


class IGCV1(nn.Module):
    """[Interleaved Group Convolutions for Deep Neural Networks](https://arxiv.org/pdf/1707.02725.pdf)"""

    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            backbone_config=L24M2_config
    ):
        super().__init__()

        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=32)

        if out_module is None:
            out_module = OutModule(output_size, input_size=10)

        self.input = in_module
        self.backbone = Backbone(backbone_config=backbone_config)
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fcn = nn.Sequential(
            Linear(self.backbone.out_channels, 10),
            out_module
        )

    def forward(self, x):
        x = self.input(x)
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fcn(x)

        return x


class Backbone(nn.Module):
    def __init__(self, backbone_config=L24M2_config):
        super().__init__()

        layers = []

        in_ch = 3
        l, m, b = backbone_config
        out_ch = l * m
        layers.append(Conv(in_ch, out_ch, 3))

        in_ch = out_ch
        for i in range(3):
            for j in range(b):
                s = 2 if i != 0 and j == 0 else 1
                layers.append(IGCBlock(in_ch, out_ch, l, s=s))
                in_ch = out_ch

            out_ch *= 2

        self.conv_seq = nn.Sequential(*layers)
        self.out_channels = in_ch

    def forward(self, x):
        return self.conv_seq(x)


class IGCBlock(nn.Module):
    def __init__(self, in_ch, out_ch, l, m=None, s=1):
        super().__init__()
        self.l = l
        self.m = m or out_ch // l

        self.c3 = Conv(in_ch, out_ch, 3, s=s)
        self.c1 = Conv(out_ch, out_ch, 1)

    def forward(self, x):
        x = self.c3(x)
        x = shuffle(x, self.l)
        x = self.c1(x)
        x = shuffle(x, self.m)
        return x


Model = IGCV1
