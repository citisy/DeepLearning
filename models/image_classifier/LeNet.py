from torch import nn
from ..layers import Conv, Linear, ConvInModule, OutModule, BaseImgClsModel


class Model(BaseImgClsModel):
    """[Handwritten Digit Recognition with a Back-Propagation Network](https://papers.nips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf)
    [Backpropagation Applied to Handwritten Zip Code Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)
    """

    def __init__(
            self,
            in_ch=None, input_size=None, out_features=None,
            out_module=None, **kwargs
    ):
        backbone = Backbone()
        neck = nn.Flatten()
        head = nn.Sequential(
            Linear(backbone.out_channels * 5 * 5, 120),
            Linear(120, 84),
            out_module or OutModule(out_features, in_features=84)
        )

        super().__init__(
            in_ch=in_ch,
            input_size=input_size,
            backbone_input_size=28,
            backbone=backbone,
            neck=neck,
            head=head,
            **kwargs
        )


class Backbone(nn.Sequential):
    def __init__(self):
        super().__init__(
            Conv(3, 6, 1, is_norm=False),
            Conv(6, 6, 3, s=2, is_norm=False),
            Conv(6, 16, 5, p=0, is_norm=False),
            Conv(16, 16, 3, s=2, is_norm=False)
        )

        self.out_channels = 16
