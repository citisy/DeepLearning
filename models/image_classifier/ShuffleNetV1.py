import torch
from torch import nn
from ..layers import Conv, Linear, ConvInModule, OutModule, BaseImgClsModel
from .MobileNetV1 import DwConv

# (groups, (out_ch, repeat))
g1_config = (1, ((144, 4), (288, 8), (576, 4)))
g2_config = (2, ((200, 4), (400, 8), (800, 4)))
g3_config = (3, ((240, 4), (480, 8), (960, 4)))
g4_config = (4, ((272, 4), (544, 8), (1088, 4)))
g8_config = (8, ((384, 4), (768, 8), (1536, 4)))


class Model(BaseImgClsModel):
    """[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083.pdf)
    See Also `torchvision.models.shufflenetv2`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, out_features=None,
            backbone_config=g3_config, **kwargs
    ):
        backbone = Backbone(backbone_config=backbone_config)
        super().__init__(
            in_ch=in_ch,
            input_size=input_size,
            out_features=out_features,
            backbone=backbone,
            **kwargs
        )


class Backbone(nn.Sequential):
    def __init__(self, backbone_config=g3_config):
        super().__init__()

        layers = []

        groups, config = backbone_config

        layers.append(Conv(3, 24, 3, s=2, groups=groups))
        layers.append(nn.MaxPool2d(3, 2))

        in_ch = 24

        for out_ch, repeat in config:
            for i in range(repeat):
                if i == 0:  # down-sample
                    layers.append(ShuffleBlock(in_ch, out_ch - in_ch, s=2, g=groups, is_concat=True))
                else:
                    layers.append(ShuffleBlock(in_ch, out_ch, g=groups))

                in_ch = out_ch

        self.conv_seq = nn.Sequential(*layers)
        self.out_channels = in_ch


class ShuffleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, s=1, g=1, is_concat=False):
        super().__init__()

        self.g = g
        mid_ch = out_ch // 4

        self.conv1 = Conv(in_ch, mid_ch, 1, groups=g)
        self.conv2 = DwConv(mid_ch, mid_ch, 3, s=s)
        self.conv3 = Conv(mid_ch, out_ch, 1, groups=g)

        self.is_concat = is_concat

        if is_concat:
            self.c = nn.AvgPool2d(3, 2, padding=1)

        self.act = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = shuffle(y, self.g)
        y = self.conv2(y)
        y = self.conv3(y)

        if self.is_concat:
            x = self.c(x)
            x = torch.cat([x, y], dim=1)
        else:
            x = x + y

        x = self.act(x)

        return x


def shuffle(x, g):
    _, c, h, w = x.size()
    c_ = c // g

    x = x.view(_, g, c_, h, w)
    # contiguous() required if transpose() is used before view()
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(_, -1, h, w)

    return x
