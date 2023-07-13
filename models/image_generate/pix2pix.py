import torch
import torch.nn as nn
from utils.layers import Conv, ConvInModule, ConvT
from utils.torch_utils import initialize_layers
from ..image_classifier.Unet import Model as NetG


class Model(nn.ModuleList):
    def __init__(self, in_ch):
        super().__init__()

        self.net_g = NetG()
        self.net_d = NetD(in_ch * 2)

        initialize_layers(self.net_g)
        initialize_layers(self.net_d)


class NetD(nn.Module):
    def __init__(self, in_ch, n_conv=3):
        super().__init__()

        out_ch = 64
        layers = [Conv(in_ch, out_ch, k=4, s=2, p=1, bias=True, act=nn.LeakyReLU(0.2, True), is_norm=False)]

        tmp_ch = out_ch
        for n in range(1, n_conv):
            in_ch = out_ch
            out_ch = tmp_ch * min(2 ** n, 8)

            layers.append(Conv(in_ch, out_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2, True)))

        in_ch = out_ch
        out_ch = tmp_ch * min(2 ** n_conv, 8)
        layers.append(Conv(in_ch, out_ch, k=4, s=1, p=1, act=nn.LeakyReLU(0.2, True)))
        layers.append(Conv(out_ch, 1, k=4, s=1, p=1, bias=True, is_norm=False, is_act=False))

        self.conv_seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_seq(x)
