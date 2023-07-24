from torch import nn
from ..layers import Conv, Linear, ConvInModule, OutModule


class AlexNet(nn.Module):
    """[ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)
    See Also `torchvision.models.alexnet`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            drop_prob=0.5, **conv_config):
        super().__init__()

        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=224)

        if out_module is None:
            out_module = OutModule(output_size, input_size=1000)

        self.input = in_module
        self.backbone = Backbone(**conv_config)
        self.flatten = nn.Flatten()
        self.fcn = nn.Sequential(
            Linear(self.backbone.out_channels * 6 * 6, 2048 * 2, is_drop=True, drop_prob=drop_prob),
            Linear(2048 * 2, 2048 * 2, is_drop=True, drop_prob=drop_prob),
            Linear(2048 * 2, 1000),
            out_module
        )

    def forward(self, x):
        x = self.input(x)
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fcn(x)

        return x


class Backbone(nn.Module):
    def __init__(self, **conv_config):
        super().__init__()
        self.conv_seq = nn.Sequential(
            Conv(3, 48 * 2, 11, s=4, p=2, **conv_config),
            nn.MaxPool2d(3, stride=2),

            Conv(48 * 2, 128 * 2, 5, **conv_config),
            nn.MaxPool2d(3, stride=2),

            Conv(128 * 2, 192 * 2, 3, **conv_config),
            Conv(192 * 2, 192 * 2, 3, **conv_config),

            Conv(192 * 2, 128 * 2, 3, **conv_config),
            nn.MaxPool2d(3, stride=2),
        )

        self.out_channels = 128 * 2

    def forward(self, x):
        return self.conv_seq(x)


Model = AlexNet
