from torch import nn
from ..layers import Conv, Linear, ConvInModule, OutModule, BaseImgClsModel


class Model(BaseImgClsModel):
    """[ImageNet Classification with Deep Convolutional Neural Networks](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf)
    See Also `torchvision.models.alexnet`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, out_features=None,
            out_module=None,
            drop_prob=0.5, conv_config=dict(), **kwargs):
        backbone = Backbone(**conv_config)
        neck = nn.Flatten()
        head = nn.Sequential(
            Linear(backbone.out_channels * 6 * 6, 2048 * 2, is_drop=True, drop_prob=drop_prob),
            Linear(2048 * 2, 2048 * 2, is_drop=True, drop_prob=drop_prob),
            Linear(2048 * 2, 1000),
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
    def __init__(self, **conv_config):
        super().__init__(
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
