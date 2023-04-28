import torch
from torch import nn
from utils.layers import Conv, Linear, ConvInModule, OutModule

# (conv_type, out_ch, k, s)
# 0: conv 1: dw
config = (
    (0, 32, 1, 2), (1, 32, 3, 1), (0, 64, 1, 1), (1, 64, 3, 2),
    (0, 128, 1, 1), (1, 128, 3, 1), (0, 128, 1, 1), (1, 128, 3, 2),
    (0, 256, 1, 1), (1, 256, 3, 1), (0, 256, 1, 1), (1, 256, 3, 2),
    (0, 512, 1, 1), *(((1, 512, 3, 1), (0, 512, 1, 1)) * 5), (1, 512, 3, 2),
    (0, 1024, 1, 1), (1, 1024, 3, 2), (0, 1024, 1, 1)
)


class MobileNetV1(nn.Module):
    """[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
    See Also `torchvision.models.mobilenet`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            conv_config=config
    ):
        super().__init__()
        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=224)

        if out_module is None:
            out_module = OutModule(output_size, input_size=1000)

        layers = []

        in_ch = 3

        for conv_type, out_ch, k, s in conv_config:
            if conv_type == 0:
                layers.append(Conv(in_ch, out_ch, k, s))
            elif conv_type == 1:
                layers.append(DwConv(in_ch, out_ch, k, s))
            else:
                raise TypeError(f'Dont support {conv_type = }')
            in_ch = out_ch

        self.input = in_module
        self.conv_seq = nn.Sequential(*layers)
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fcn = nn.Sequential(
            Linear(1 * 1 * in_ch, 1000),
            out_module
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv_seq(x)
        x = self.flatten(x)
        x = self.fcn(x)

        return x


class DwConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, act=nn.ReLU6):
        super().__init__()

        self.conv_seq = nn.Sequential(
            Conv(in_ch, in_ch, k, s, conv_kwargs=dict(groups=in_ch), act=act),
            Conv(in_ch, out_ch, 1, act=act)
        )

    def forward(self, x):
        return self.conv_seq(x)


Model = MobileNetV1
