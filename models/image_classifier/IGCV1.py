from torch import nn
from ..layers import Conv, Linear, ConvInModule, OutModule, BaseImgClsModel
from .ShuffleNetV1 import shuffle

# (l, m, b)
L4M8_config = (4, 8, 6)
L24M2_config = (24, 2, 6)
L32M26_config = (32, 26, 6)


class Model(BaseImgClsModel):
    """[Interleaved Group Convolutions for Deep Neural Networks](https://arxiv.org/pdf/1707.02725.pdf)"""

    def __init__(
            self,
            in_ch=None, input_size=None, out_features=None,
            backbone_config=L24M2_config, **kwargs
    ):
        backbone = Backbone(backbone_config=backbone_config)

        super().__init__(
            in_ch=in_ch,
            input_size=input_size,
            out_features=out_features,
            backbone=backbone,
            backbone_input_size=32,
            head_hidden_features=10,
            **kwargs
        )


class Backbone(nn.Sequential):
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
