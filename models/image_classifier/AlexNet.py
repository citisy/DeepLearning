from torch import nn
from utils.layers import Conv, Linear, ConvInModule, OutModule


class AlexNet(nn.Module):
    """[ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)
    See Also `torchvision.models.alexnet`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            drop_prob=0.5):
        super().__init__()

        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=224)

        if out_module is None:
            out_module = OutModule(output_size, input_size=1000)

        self.input = in_module
        self.conv_seq = nn.Sequential(
            in_module,

            Conv(3, 48 * 2, 11, s=4, p=2),
            nn.MaxPool2d(3, stride=2),

            Conv(48 * 2, 128 * 2, 5),
            nn.MaxPool2d(3, stride=2),

            Conv(128 * 2, 192 * 2, 3),
            Conv(192 * 2, 192 * 2, 3),

            Conv(192 * 2, 128 * 2, 3),
            nn.MaxPool2d(3, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fcn = nn.Sequential(
            Linear(128 * 2 * 6 * 6, 2048 * 2),
            nn.Dropout(drop_prob),

            Linear(2048 * 2, 2048 * 2),
            nn.Dropout(drop_prob),

            Linear(2048 * 2, 1000),
            out_module
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv_seq(x)
        x = self.flatten(x)
        x = self.fcn(x)

        return x


Model = AlexNet
