import torch
import torch.nn as nn
import numpy as np
from ..layers import Conv, ConvT
from utils.torch_utils import initialize_layers


class Model(nn.ModuleList):
    """refer to
    paper:
        - [Towards Principled Methods for Training Generative Adversarial Networks](https://arxiv.org/pdf/1701.04862.pdf)
        - [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf)
    code:
        - https://github.com/martinarjovsky/WassersteinGAN
    """
    def __init__(self, input_size, in_ch, hidden_ch,
                 net_g=None, net_d=None,
                 n_conv=0):
        super().__init__()

        self.net_d = net_d if net_d is not None else DcganD(input_size, in_ch, n_conv)
        self.net_g = net_g if net_g is not None else DcganG(input_size, hidden_ch, in_ch, n_conv)

        initialize_layers(self)

        self.hidden_ch = hidden_ch

    def gen_noise(self, batch_size, device):
        return torch.normal(mean=0., std=1., size=(batch_size, self.hidden_ch, 1, 1), device=device)

    def loss_d(self, real_x):
        self.net_d.requires_grad_(True)

        # note that, weight clipping, if change to gradient penalty, it is WGAN-GP
        # see also https://github.com/martinarjovsky/WassersteinGAN/issues/18
        for p in self.net_d.parameters():
            p.data.clamp_(-0.01, 0.01)

        self.net_d.zero_grad()

        # note that, pred output backward without loss function,
        # see also https://github.com/martinarjovsky/WassersteinGAN/issues/9
        # 1. real_x -> net_d -> pred_real -> loss_d_real -> gradient_descent
        # loss_d_real = net_d(real_x)
        pred_real = self.net_d(real_x)
        loss_d_real = pred_real

        # 2. noise -> net_g -> fake_x -> net_d -> pred_fake -> loss_d_fake -> gradient_ascent
        # loss_d_fake = net_d(net_g(noise))
        with torch.no_grad():
            noise = self.gen_noise(len(real_x), real_x.device)
            fake_x = self.net_g(noise)

        pred_fake = self.net_d(fake_x)
        loss_d_fake = -pred_fake
        loss_d = loss_d_real + loss_d_fake
        return loss_d

    def loss_g(self, real_x):
        self.net_d.requires_grad_(False)
        self.net_g.zero_grad()

        # 1. noise -> net_g -> fake_x -> net_d -> pred_fake -> loss_g -> gradient_descent
        # loss_g = net_d(net_g(noise))
        noise = self.gen_noise(len(real_x), real_x.device)
        fake_x = self.net_g(noise)
        loss_g = self.net_d(fake_x)
        return loss_g


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
