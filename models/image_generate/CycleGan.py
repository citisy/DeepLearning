import torch
import torch.nn as nn
from utils.layers import Conv, ConvInModule, ConvT
from utils.torch_utils import initialize_layers
from .pix2pix import NetD, net_d_config
from ..image_classifier.ResNet import ResBlock

net_g_config = dict(
    n_res_blocks=9
)


class Model(nn.ModuleList):
    """refer to:
    paper:
        [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)
    code:
        - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    def __init__(self, in_ch, input_size, net_g_config=net_g_config, net_d_config=net_d_config):
        super().__init__()

        self.net_g_a = NetG(in_ch, 3, **net_g_config)
        self.net_g_b = NetG(in_ch, 3, **net_g_config)
        self.net_d_a = NetD(in_ch, **net_d_config)
        self.net_d_b = NetD(in_ch, **net_d_config)

        initialize_layers(self)


class NetG(nn.Module):
    def __init__(self, in_ch, output_size, hidden_ch=64, n_sample_conv=2, n_res_blocks=9, ):
        super().__init__()

        out_ch = hidden_ch
        layers = [Conv(in_ch, out_ch, k=7, p=3)]

        # down_sampling
        for n in range(n_sample_conv):
            in_ch = out_ch
            out_ch *= 2
            layers.append(Conv(in_ch, out_ch, k=3, s=2, p=1))

        # res block
        in_ch = out_ch
        for n in range(n_res_blocks):
            layers.append(ResBlock(in_ch, in_ch))

        # up_sampling
        for n in range(n_sample_conv):
            in_ch = out_ch
            out_ch //= 2
            layers.append(ConvT(in_ch, out_ch, k=3, s=2, p=1, output_padding=1))

        in_ch = out_ch
        layers.append(Conv(in_ch, output_size, k=7, p=3, act=nn.Tanh(), is_norm=False))

        self.conv_seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_seq(x)
