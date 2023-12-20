from torch import nn
from ..layers import Linear
from .InceptionV1 import Model as Inception, InceptionA, Inception_config, Backbone as _Backbone


class Model(Inception):
    """[Squeeze-and-Excitation Networks](https://arxiv.org/pdf/1709.01507.pdf)

    note that, don't use InceptionV2 which using batch normalization, it is not work for SENet
    """

    def __init__(
            self,
            in_ch=None, input_size=None, out_features=None,
            backbone_config=Inception_config, **kwargs):
        super().__init__(in_ch, input_size, out_features, backbone=Backbone,
                         backbone_config=backbone_config, **kwargs)


class Backbone(_Backbone):
    def __init__(self, backbone_config=Inception_config, **kwargs):
        super().__init__(backbone_config=backbone_config, **kwargs)

        # 'Sequential' object has no attribute 'insert' where torch.version < 1.12.x
        j = 1
        for i, module in list(self.conv_seq.named_children()):
            if isinstance(module, InceptionA):
                self.conv_seq.insert(int(i) + j, SEBlock(module.out_ch))
                j += 1


class SEBlock(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()

        self.sq = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.ex = nn.Sequential(
            Linear(ch, ch // r, act=nn.ReLU(), is_norm=False),
            Linear(ch // r, ch, is_norm=False)
        )

    def forward(self, x):
        y = self.sq(x)
        y = self.ex(y)
        y = y.view(*x.size()[:2], 1, 1).expand_as(x)
        x = x * y

        return x

