import torch
from torch import nn
from utils.layers import Conv, Linear, ConvInModule, OutModule

# (sq_ch, ex1_ch, ex3_ch)
config = (
    ((16, 64, 64), (16, 64, 64), (32, 128, 128)),
    ((32, 128, 128), (48, 192, 192), (48, 192, 192), (64, 256, 256)),
    ((64, 256, 256),)
)


class SqueezeNet(nn.Module):
    """[Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size](https://arxiv.org/pdf/1602.07360.pdf)
    See Also `torchvision.models.squeezenet`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            conv_config=config):
        super().__init__()
        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=224)

        if out_module is None:
            out_module = OutModule(output_size, input_size=1000)

        layers = [
            in_module,
            Conv(3, 96, 7, s=2),
        ]

        in_ch = 96

        for a in conv_config:
            layers.append(nn.MaxPool2d(3, stride=2))
            for out_ches in a:
                layers.append(Fire(in_ch, *out_ches))
                in_ch = out_ches[1] + out_ches[2]

        layers.append(Conv(in_ch, 1000, 1))
        layers.append(nn.AdaptiveAvgPool2d(1))

        self.conv_seq = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fcn = nn.Sequential(
            Linear(1 * 1 * 1000, 1000),
            out_module
        )

    def forward(self, x):
        x = self.conv_seq(x)
        x = self.flatten(x)
        x = self.fcn(x)

        return x


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


Model = SqueezeNet
