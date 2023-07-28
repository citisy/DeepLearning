from torch import nn
from ..layers import Conv, Linear, ConvInModule, OutModule, BaseImgClsModel

# (conv_type, out_ch, k, s)
# 0: conv 1: dw
default_config = (
    (0, 32, 1, 2), (1, 32, 3, 1), (0, 64, 1, 1), (1, 64, 3, 2),
    (0, 128, 1, 1), (1, 128, 3, 1), (0, 128, 1, 1), (1, 128, 3, 2),
    (0, 256, 1, 1), (1, 256, 3, 1), (0, 256, 1, 1), (1, 256, 3, 2),
    (0, 512, 1, 1), *(((1, 512, 3, 1), (0, 512, 1, 1)) * 5), (1, 512, 3, 2),
    (0, 1024, 1, 1), (1, 1024, 3, 2), (0, 1024, 1, 1)
)


class Model(BaseImgClsModel):
    """[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
    See Also `torchvision.models.mobilenet`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, out_features=None,
            backbone_config=default_config, conv_config=dict(), **kwargs
    ):
        backbone = Backbone(backbone_config=backbone_config, **conv_config)

        super().__init__(
            in_ch=in_ch,
            input_size=input_size,
            out_features=out_features,
            backbone=backbone,
            **kwargs
        )


class Backbone(nn.Sequential):
    def __init__(self, backbone_config=default_config, **conv_config):
        layers = []

        in_ch = 3

        for conv_type, out_ch, k, s in backbone_config:
            if conv_type == 0:
                layers.append(Conv(in_ch, out_ch, k, s, **conv_config))
            elif conv_type == 1:
                layers.append(DwConv(in_ch, out_ch, k, s, **conv_config))
            else:
                raise TypeError(f'Dont support {conv_type = }')
            in_ch = out_ch

        self.out_channels = in_ch
        super().__init__(*layers)


class DwConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, act=None):
        super().__init__(
            Conv(in_ch, in_ch, k, s, groups=in_ch, act=act or nn.ReLU6(True)),
            Conv(in_ch, out_ch, 1, act=act or nn.ReLU6(True))
        )
