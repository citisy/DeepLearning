from torch import nn
from utils.layers import Conv, Linear, ConvInModule, OutModule

# refer to table 1
# (n_conv, out_ch)
VGGA_config = VGG11_config = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  # 11 = 8 conv + 3 fcn
VGGB_config = VGG13_config = ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512))
VGGD_config = VGG16_config = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
VGGE_config = VGG19_config = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))


class VGG(nn.Module):
    """[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
    See Also `torchvision.models.vgg`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            conv_config=VGG11_config, drop_prob=0.5):
        super().__init__()

        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=224)

        if out_module is None:
            out_module = OutModule(output_size, input_size=1000)

        layers = []

        in_ch, out_ch = 3, 3

        for n_conv, out_ch in conv_config:
            layers.append(VGGBlock(in_ch, out_ch, n_conv))
            in_ch = out_ch

        self.input = in_module
        self.conv_seq = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fcn = nn.Sequential(
            Linear(out_ch * 7 * 7, 4096, is_drop=True, drop_prob=drop_prob),  # 7 = 224/2^5
            Linear(4096, 4096, is_drop=True, drop_prob=drop_prob),
            Linear(4096, 1000),
            out_module
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv_seq(x)
        x = self.flatten(x)
        x = self.fcn(x)

        return x


class VGGBlock(nn.Module):
    def __init__(self, in_ch, out_ch, n_conv):
        super().__init__()

        layers = []

        for _ in range(n_conv):
            layers.append(Conv(in_ch, out_ch, 3))
            in_ch = out_ch

        self.conv_block = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.pool(x)
        return x


Model = VGG
