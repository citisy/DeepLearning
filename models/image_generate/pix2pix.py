import torch
import torch.nn as nn
from utils.layers import Conv, ConvInModule, ConvT
from utils.torch_utils import initialize_layers
from ..image_classifier.Unet import Model as NetG, unet256_config

net_g_config = dict(
    conv_config=unet256_config
)

net_d_config = dict(
    hidden_ch=64,
    n_conv=3,
)


class Model(nn.ModuleList):
    def __init__(self, in_ch, input_size, net_g_config=net_g_config, net_d_config=net_d_config):
        super().__init__()

        self.net_g = NetG(in_ch, input_size, **net_g_config)
        self.net_d = NetD(in_ch * 2, **net_d_config)

        initialize_layers(self.net_g)
        initialize_layers(self.net_d)


class NetD(nn.Module):
    def __init__(self, in_ch, hidden_ch=64, n_conv=3):
        super().__init__()

        layers = [Conv(in_ch, hidden_ch, k=4, s=2, p=1, bias=True, act=nn.LeakyReLU(0.2, True), is_norm=False)]

        out_ch = hidden_ch
        for n in range(1, n_conv + 1):
            in_ch = out_ch
            out_ch = hidden_ch * min(2 ** n, 8)

            if n == n_conv:
                layers.append(Conv(in_ch, out_ch, k=4, s=1, p=1, act=nn.LeakyReLU(0.2, True)))
            else:
                layers.append(Conv(in_ch, out_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2, True)))

        layers.append(Conv(out_ch, 1, k=4, s=1, p=1, bias=True, is_norm=False, is_act=False))
        self.conv_seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_seq(x)
