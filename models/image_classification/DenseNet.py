import torch
from torch import nn
from . import BaseImgClsModel
from ..layers import Conv, Linear, ConvInModule, OutModule

# refer to table 1
# (n_conv, growth_rate)
Dense121_config = ((6, 32), (12, 32), (24, 32), (16, 32))  # 121 = 1 conv + (6 + 12 + 24 + 16) * 2 conv + 3 conv + 1 fcn
Dense169_config = ((6, 32), (12, 32), (32, 32), (32, 32))
Res201_config = ((6, 32), (12, 32), (48, 32), (32, 32))
Res264_config = ((6, 32), (12, 32), (64, 32), (48, 32))


class Model(BaseImgClsModel):
    """[Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
    See Also `torchvision.models.densenet`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, out_features=None,
            backbone=None, backbone_config=Dense121_config, block=None, **kwargs
    ):
        backbone = backbone or Backbone(backbone_config=backbone_config, block=block)

        super().__init__(
            in_ch=in_ch,
            input_size=input_size,
            out_features=out_features,
            backbone=backbone,
            **kwargs
        )


class Backbone(nn.Sequential):
    def __init__(self, backbone_config=Dense121_config, block=None):
        block = block or DenseBlock

        layers = [
            Conv(3, 64, 7, s=2),
            nn.MaxPool2d(3, stride=2, padding=1)
        ]

        in_ch = 64

        for i, (n_conv, growth_rate) in enumerate(backbone_config):
            layers.append(block(in_ch, growth_rate, n_conv))

            in_ch += growth_rate * n_conv

            if i < len(backbone_config) - 1:
                out_ch = in_ch // 2
                layers.append(Transition(in_ch, out_ch))
                in_ch = out_ch

        self.out_channels = in_ch
        super().__init__(*layers)


class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth_rate, n_conv=2):
        super().__init__()

        self.in_channels = in_ch
        self.growth_rate = growth_rate
        self.n_conv = n_conv

        layers = []

        for _ in range(n_conv):
            layers.append(nn.Sequential(
                Conv(in_ch, 4 * growth_rate, k=1),
                Conv(4 * growth_rate, growth_rate, k=3)
            ))
            in_ch += growth_rate

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for conv in self.layers:
            xl = conv(x)
            x = torch.cat((x, xl), dim=1)

        return x


class Transition(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            Conv(in_ch, out_ch, 1),
            nn.AvgPool2d(2, 2)
        )
