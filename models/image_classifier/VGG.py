from torch import nn
from ..layers import Conv, Linear, ConvInModule, OutModule, BaseImgClsModel

# refer to table 1
# (n_conv, out_ch)
VGGA_config = VGG11_config = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  # 11 = 8 conv + 3 fcn
VGGB_config = VGG13_config = ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512))
VGGD_config = VGG16_config = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
VGGE_config = VGG19_config = ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))


class Model(BaseImgClsModel):
    """[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf)
    See Also `torchvision.models.vgg`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, out_features=None,
            out_module=None,
            backbone_config=VGG11_config, drop_prob=0.5, conv_config=dict(), **kwargs
    ):
        backbone = Backbone(backbone_config=backbone_config, **conv_config)
        neck = nn.Flatten()
        head = nn.Sequential(
            Linear(backbone.out_channels * 7 * 7, 4096, is_drop=True, drop_prob=drop_prob),  # 7 = 224/2^5
            Linear(4096, 4096, is_drop=True, drop_prob=drop_prob),
            Linear(4096, 1000),
            out_module or OutModule(out_features, in_features=1000)
        )

        super().__init__(
            in_ch=in_ch,
            input_size=input_size,
            backbone=backbone,
            neck=neck,
            head=head,
            **kwargs
        )


class Backbone(nn.Sequential):
    def __init__(self, backbone_config=VGG11_config, **conv_config):
        super().__init__()
        layers = []
        in_ch, out_ch = 3, 3

        for n_conv, out_ch in backbone_config:
            layers.append(VGGBlock(in_ch, out_ch, n_conv, **conv_config))
            in_ch = out_ch

        self.conv_seq = nn.Sequential(*layers)
        self.out_channels = in_ch


class VGGBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, n_conv, **conv_config):
        super().__init__()

        layers = []

        for _ in range(n_conv):
            layers.append(Conv(in_ch, out_ch, 3, **conv_config))
            in_ch = out_ch

        self.conv_block = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2)
