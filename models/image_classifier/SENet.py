from torch import nn
from utils.layers import Conv, Linear
from .Inception import InceptionV1, InceptionA, InceptionV1_config
from .ResNet import ResNet, Res18_config


class SEInception(InceptionV1):
    """[Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)"""

    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            conv_config=InceptionV1_config, drop_prob=0.4):
        super().__init__(in_ch, input_size, output_size, in_module, out_module, conv_config, drop_prob)

        j = 1
        for i, module in list(self.conv_seq._modules.items()):
            if isinstance(module, InceptionA):
                self.conv_seq.insert(int(i) + j, SEBlock(module.out_ch))
                j += 1


class SEResNet(ResNet):
    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            conv_config=Res18_config
    ):
        super().__init__(
            in_ch, input_size, output_size,
            in_module, out_module,
            conv_config,
            add_block=SEBlock,
        )


class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()

        self.sq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.ex = nn.Sequential(
            Linear(ch, ch // r, act=nn.ReLU, is_bn=False),
            Linear(ch // r, ch, is_bn=False)
        )

    def forward(self, x):
        y = self.sq(x)
        y = self.ex(y)
        y = y.view(*x.size()[:2], 1, 1).expand_as(x)
        x = x * y

        return x


Model = SEInception
