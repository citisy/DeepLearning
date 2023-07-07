import torch
from torch import nn
from utils.layers import Conv, Linear, ConvInModule, OutModule

# refer to table 1
# (n_conv, growth_rate)
Dense121_config = ((6, 32), (12, 32), (24, 32), (16, 32))  # 121 = 1 conv + (6 + 12 + 24 + 16) * 2 conv + 3 conv + 1 fcn
Dense169_config = ((6, 32), (12, 32), (32, 32), (32, 32))
Res201_config = ((6, 32), (12, 32), (48, 32), (32, 32))
Res264_config = ((6, 32), (12, 32), (64, 32), (48, 32))


class DenseNet(nn.Module):
    """[Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
    See Also `torchvision.models.densenet`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            conv_config=Dense121_config
    ):
        super().__init__()
        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=224)

        if out_module is None:
            out_module = OutModule(output_size, input_size=1000)

        layers = [
            Conv(3, 64, 7, s=2),
            nn.MaxPool2d(3, stride=2, padding=1)
        ]

        in_ch = 64

        for i, (n_conv, growth_rate) in enumerate(conv_config):
            layers.append(DenseBlock(in_ch, growth_rate, n_conv))

            in_ch += growth_rate * n_conv

            if i < len(conv_config) - 1:
                out_ch = in_ch // 2
                layers.append(Transition(in_ch, out_ch))
                in_ch = out_ch

        self.input = in_module
        self.conv_seq = nn.Sequential(*layers)
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fcn = nn.Sequential(
            Linear(in_ch, 1000),
            out_module
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv_seq(x)
        x = self.flatten(x)
        x = self.fcn(x)

        return x


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

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for conv in self.layers:
            xl = conv(x)
            x = torch.cat((x, xl), dim=1)

        return x


class Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            Conv(in_ch, out_ch, 1),
            nn.AvgPool2d(2, 2)
        )

    def forward(self, x):
        return self.block(x)


Model = DenseNet
