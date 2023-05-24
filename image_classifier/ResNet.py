import torch
from torch import nn
from utils.layers import Conv, Linear, ConvInModule, OutModule

# refer to table 1
# (n_res, out_ch, n_conv)
Res18_config = ((2, 64, 2), (2, 128, 2), (2, 256, 2), (2, 512, 2))  # 18 = 1 conv + 16 conv + 1 fcn
Res34_config = ((3, 64, 2), (4, 128, 2), (6, 256, 2), (3, 512, 2))
Res50_config = ((3, 256, 3), (4, 512, 3), (6, 1024, 3), (3, 2048, 3))
Res101_config = ((3, 256, 3), (4, 512, 3), (23, 1024, 3), (3, 2048, 3))
Res152_config = ((3, 256, 3), (8, 512, 3), (36, 1024, 3), (3, 2048, 3))


class ResNet(nn.Module):
    """[Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
    See Also `torchvision.models.resnet`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            conv_config=Res18_config,
            add_block: nn.Module = None, block_config=dict()):
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

        for i, (n_res, out_ch, n_conv) in enumerate(conv_config):
            for j in range(n_res):
                if i != 0 and j == 0:
                    layers.append(ResBlock(in_ch, out_ch, n_conv, s=2, add_block=add_block, block_config=block_config))
                else:
                    layers.append(ResBlock(in_ch, out_ch, n_conv, add_block=add_block, block_config=block_config))

                in_ch = out_ch

        self.input = in_module
        self.conv_seq = nn.Sequential(*layers)
        self.conv_seq.out_channels = in_ch
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fcn = nn.Sequential(
            Linear(1 * 1 * 512, 1000),
            out_module
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv_seq(x)
        x = self.flatten(x)
        x = self.fcn(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_conv=2, s=1,
                 add_block: nn.Module = None, block_config=dict()):
        super().__init__()
        if n_conv == 2:
            self.conv_seq = nn.Sequential(
                Conv(in_ch, out_ch, k=3, s=s),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch)
            )
        elif n_conv == 3:  # use bottleneck
            tmp_ch = out_ch // 4
            self.conv_seq = nn.Sequential(
                Conv(in_ch, tmp_ch, k=1, s=s),
                Conv(tmp_ch, tmp_ch, k=3),
                nn.Conv2d(tmp_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_ch)
            )

        else:
            raise ValueError(f'Not supported {n_conv = }')

        if add_block:
            self.conv_seq.append(add_block(out_ch, **block_config))

        self.is_projection_x = in_ch != out_ch
        if self.is_projection_x:
            self.conv_x = Conv(in_ch, out_ch, k=1, s=s)

        self.act = nn.ReLU()

    def forward(self, x):
        xl = self.conv_seq(x)

        if self.is_projection_x:
            x = self.conv_x(x)

        x = self.act(xl + x)
        return x


Model = ResNet
