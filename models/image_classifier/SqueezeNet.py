import torch
from torch import nn
from . import BaseImgClsModel
from ..layers import Conv, Linear, ConvInModule, OutModule

# (sq_ch, ex1_ch, ex3_ch)
default_config = (
    ((16, 64, 64), (16, 64, 64), (32, 128, 128)),
    ((32, 128, 128), (48, 192, 192), (48, 192, 192), (64, 256, 256)),
    ((64, 256, 256),)
)


class Model(BaseImgClsModel):
    """[Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size](https://arxiv.org/pdf/1602.07360.pdf)
    See Also `torchvision.models.squeezenet`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, out_features=None,
            backbone_config=default_config, **kwargs):
        backbone = Backbone(backbone_config=backbone_config)

        super().__init__(
            in_ch=in_ch,
            input_size=input_size,
            out_features=out_features,
            backbone=backbone,
            **kwargs
        )


class Backbone(nn.Sequential):
    def __init__(self, backbone_config=default_config):
        layers = [
            Conv(3, 96, 7, s=2),
        ]

        in_ch = 96

        for a in backbone_config:
            layers.append(nn.MaxPool2d(3, stride=2))
            for out_ches in a:
                layers.append(Fire(in_ch, *out_ches))
                in_ch = out_ches[1] + out_ches[2]

        layers.append(Conv(in_ch, 1000, 1))

        self.out_channels = 1000
        super().__init__(*layers)


class Fire(nn.Module):
    def __init__(self, in_ch, *out_ches):
        super().__init__()

        self.sq = Conv(in_ch, out_ches[0], 1)
        self.ex1 = Conv(out_ches[0], out_ches[1], 1)
        self.ex3 = Conv(out_ches[0], out_ches[2], 3)

    def forward(self, x):
        x = self.sq(x)
        x1 = self.ex1(x)
        x3 = self.ex3(x)
        x = torch.cat([x1, x3], dim=1)
        return x
