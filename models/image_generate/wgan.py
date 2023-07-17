import torch
import torch.nn as nn
import numpy as np
from utils.layers import Conv, ConvInModule, ConvT
from utils.torch_utils import initialize_layers


class Model(nn.ModuleList):
    """refer to
    paper:
        - [Towards Principled Methods for Training Generative Adversarial Networks](https://arxiv.org/pdf/1701.04862.pdf)
        - [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)
    code:
        - https://github.com/martinarjovsky/WassersteinGAN
    """
    def __init__(self, input_size, in_ch, hidden_ch, n_conv=0):
        super().__init__()

        self.net_d = DcganD(input_size, in_ch, n_conv)
        self.net_g = DcganG(input_size, hidden_ch, in_ch, n_conv)

        initialize_layers(self)

        self.hidden_ch = hidden_ch


class DcganD(nn.Module):
    def __init__(self, input_size, in_ch, n_conv=0):
        super().__init__()
        assert input_size % 16 == 0, "input_size has to be a multiple of 16"

        out_ch = 64

        layers = [Conv(in_ch, out_ch, 4, s=2, p=1, is_norm=False, act=nn.LeakyReLU(0.2))]
        in_ch = out_ch

        for _ in range(n_conv):
            layers.append(Conv(in_ch, in_ch, 3, s=1, p=1, act=nn.LeakyReLU(0.2)))

        n = np.log2(input_size // 8).astype(int)
        for _ in range(n):
            in_ch = out_ch
            out_ch *= 2
            layers.append(Conv(in_ch, out_ch, 4, s=2, p=1, act=nn.LeakyReLU(0.2)))

        in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, 1, 4, 1, 0, bias=False))

        self.conv_seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_seq(x).mean(0).view(1)


class DcganG(nn.Module):
    def __init__(self, input_size, in_ch, output_size, n_conv=0):
        super().__init__()
        assert input_size % 16 == 0, "input_size has to be a multiple of 16"

        out_ch = 32 * input_size // 4

        layers = [ConvT(in_ch, out_ch, 4, s=1, p=0)]

        n = np.log2(input_size // 8).astype(int)
        for _ in range(n):
            in_ch = out_ch
            out_ch //= 2
            layers.append(ConvT(in_ch, out_ch, 4, s=2, p=1))

        in_ch = out_ch
        for _ in range(n_conv):
            layers.append(Conv(in_ch, in_ch, 3, s=1, p=1))

        layers.append(ConvT(in_ch, output_size, 4, s=2, p=1, is_norm=False, act=nn.Tanh()))
        self.conv_seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_seq(x)
