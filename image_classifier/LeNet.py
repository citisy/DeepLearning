from torch import nn
from utils.layers import Conv, Linear, ConvInModule, OutModule


class Model(nn.Module):
    """[Handwritten Digit Recognition with a Back-Propagation Network](https://papers.nips.cc/paper/1989/file/53c3bce66e43be4f209556518c2fcb54-Paper.pdf)
    [Backpropagation Applied to Handwritten Zip Code Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-89e.pdf)
    """

    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
    ):
        super().__init__()
        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=28)

        if out_module is None:
            out_module = OutModule(output_size, input_size=84)

        self.conv_seq = nn.Sequential(
            in_module,
            Conv(3, 6, 1),
            Conv(6, 6, 3, s=2),
            Conv(6, 16, 5, p=0),
            Conv(16, 16, 3, s=2)
        )
        self.flatten = nn.Flatten()
        self.fcn = nn.Sequential(
            Linear(16 * 5 * 5, 120),
            Linear(120, 84),
            out_module
        )

    def forward(self, x):
        x = self.conv_seq(x)
        x = self.flatten(x)
        x = self.fcn(x)

        return x
