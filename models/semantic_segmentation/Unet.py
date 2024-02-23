import torch
import torch.nn as nn
from . import BaseSemSegModel
from ..layers import Conv, ConvInModule, ConvT
from utils.torch_utils import ModuleManager


class Config:
    unet256 = dict(
        unit_ch=64,
        ch_mult=(1, 2, 4, 8, 8, 8, 8, 8)   # the last stage is mid-layer
    )

    @classmethod
    def get(cls, name='unet256'):
        return dict(
            backbone_config=getattr(cls, name)
        )


class Model(BaseSemSegModel):
    """refer to [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)"""

    def __init__(self, in_ch, out_features, backbone_config=Config.unet256, **kwargs):
        super().__init__()

        self.out_features = out_features + 1

        # top(outer) -> bottom(inner)
        # self.backbone = CurUnetBlock(in_ch, self.out_features, **backbone_config)
        self.backbone = CirUnetBlock(in_ch, self.out_features, **backbone_config)
        ModuleManager.initialize_layers(self)

    def forward(self, x, pix_images=None):
        x = self.backbone(x)
        return super().forward(x, pix_images)


class CurUnetBlock(nn.Module):
    """Unet block built by recursion strategy"""

    def __init__(self, in_ch, out_ch, unit_ch=64, ch_mult=(1, 2, 4, 8, 8, 8, 8, 8), layer_idx=0):
        super().__init__()
        self.is_bottom = layer_idx == len(ch_mult) - 1
        self.is_top = layer_idx == 0
        self.layer_idx = layer_idx

        layers = []
        mult = ch_mult[layer_idx]
        hidden_ch = unit_ch * mult

        # down
        layers.append(self.make_down_layer(in_ch, hidden_ch, self.is_top, self.is_bottom))

        if self.is_bottom:
            # mid
            layers.append(self.make_mid_layer(in_ch, out_ch))

        else:
            layers.append(CurUnetBlock(
                hidden_ch, hidden_ch, unit_ch, ch_mult,
                layer_idx=layer_idx + 1,
            ))

        # up
        layers.append(self.make_up_layer(hidden_ch, out_ch, self.is_top, self.is_bottom))

        self.conv_seq = nn.Sequential(*layers)

    def make_down_layer(self, in_ch, out_ch, is_top=False, is_bottom=False):
        if is_top:
            layer = Conv(in_ch, out_ch, k=4, s=2, p=1, mode='c')
        elif is_bottom:
            layer = nn.Identity()
        else:
            layer = Conv(in_ch, out_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2), mode='acn')

        return layer

    def make_mid_layer(self, in_ch, out_ch):
        return nn.Sequential(
            Conv(in_ch, out_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2), mode='ac'),
            ConvT(out_ch, out_ch, k=4, s=2, p=1, mode='acn')
        )

    def make_up_layer(self, in_ch, out_ch, is_top=False, is_bottom=False):
        if is_top:
            layer = ConvT(in_ch * 2, out_ch, k=4, s=2, p=1, mode='ac')
        elif is_bottom:
            layer = nn.Identity()
        else:
            layer = ConvT(in_ch * 2, out_ch, k=4, s=2, p=1, mode='acn')

        return layer

    def forward(self, x):
        y = self.conv_seq(x)
        if self.is_top:
            return y
        else:
            return torch.cat([x, y], 1)


class CirUnetBlock(nn.Module):
    """Unet block built by cyclic strategy"""

    def __init__(self, in_ch, out_ch, unit_ch=64, ch_mult=(1, 2, 4, 8, 8, 8, 8, 8)):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch

        num_stages = len(ch_mult) - 1   # the last stage is mid-layer

        layers = []
        for i, mult in enumerate(ch_mult[:-1]):
            is_top = i == 0
            is_bottom = i == num_stages - 1

            out_ch = unit_ch * mult
            layers.append(self.make_down_layer(in_ch, out_ch, is_top, is_bottom))
            in_ch = out_ch

        self.downs = nn.ModuleList(layers)

        out_ch = unit_ch * ch_mult[-1]
        self.mid = self.make_mid_layer(in_ch, in_ch)
        in_ch = out_ch

        layers = []
        for i in reversed(range(num_stages)):
            is_top = i == 0
            is_bottom = i == num_stages - 1

            out_ch = unit_ch * ch_mult[i - 1] if not is_top else self.out_channels
            layers.append(self.make_up_layer(in_ch, out_ch, is_top, is_bottom))
            in_ch = out_ch

        self.ups = nn.ModuleList(layers)

    def make_down_layer(self, in_ch, out_ch, is_top=False, is_bottom=False):
        if is_top:
            layer = Conv(in_ch, out_ch, k=4, s=2, p=1, mode='c')
        else:
            layer = Conv(in_ch, out_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2), mode='acn')

        return layer

    def make_mid_layer(self, in_ch, out_ch):
        return nn.Sequential(
            Conv(in_ch, out_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2), mode='ac'),
            ConvT(out_ch, in_ch, k=4, s=2, p=1, mode='acn')
        )

    def make_up_layer(self, in_ch, out_ch, is_top=False, is_bottom=False):
        if is_top:
            layer = ConvT(in_ch * 2, out_ch, k=4, s=2, p=1, mode='ac')
        else:
            layer = ConvT(in_ch * 2, out_ch, k=4, s=2, p=1, mode='acn')

        return layer

    def forward(self, x):
        hs = []
        for down in self.downs:
            x = down(x)
            hs.append(x)

        x = self.mid(x)

        for up in self.ups:
            h = hs.pop()
            x = torch.cat([x, h], 1)
            x = up(x)

        return x
