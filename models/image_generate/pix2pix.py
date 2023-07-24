import torch
import torch.nn as nn
from .. import Conv
from utils.torch_utils import initialize_layers
from ..semantic_segmentation.Unet import Model as NetG, unet256_config

net_g_config = dict(
    conv_config=unet256_config
)

net_d_config = dict(
    hidden_ch=64,
    n_conv=3,
)


class Model(nn.ModuleList):
    """refer to:
    paper:
        - [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
    code:
        - https://github.com/phillipi/pix2pix
        - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, in_ch, input_size, net_g_config=net_g_config, net_d_config=net_d_config,
                 lambda_l1=100.0, real_label=1., fake_label=0.,
                 **kwargs):
        super().__init__()

        self.net_g = NetG(in_ch, input_size, **net_g_config)
        self.net_d = NetD(in_ch * 2, **net_d_config)

        initialize_layers(self.net_g)
        initialize_layers(self.net_d)

        self.gan_loss_fn = nn.MSELoss()
        self.l1_loss_fn = torch.nn.L1Loss()
        self.lambda_l1 = lambda_l1
        self.real_label = torch.tensor(real_label)
        self.fake_label = torch.tensor(fake_label)

    def loss_d(self, real_a, real_b, fake_ab):
        self.net_d.requires_grad_(True)

        # 1. real_a + fake_b -> net_d -> pred_fake, fake_label -> loss_d_fake
        # loss_d_fake = net_d(real_a, fake_b) - 0
        pred_fake = self.net_d(fake_ab.detach())
        loss_d_fake = self.gan_loss_fn(pred_fake, self.fake_label.to(pred_fake).expand_as(pred_fake))

        # 2. real_a + real_b -> net_d -> pred_real, real_label -> loss_d_real
        # loss_d_real = net_d(real_a, real_b) - 1
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = self.net_d(real_ab)
        loss_d_real = self.gan_loss_fn(pred_real, self.real_label.to(pred_real).expand_as(pred_real))

        loss_d = (loss_d_fake + loss_d_real) * 0.5

        return loss_d

    def loss_g(self, real_b, fake_b, fake_ab):
        # update G
        self.net_d.requires_grad_(False)

        #  1. real_a + fake_b -> net_d -> pred_fake, real_label -> loss_g_gan
        # loss_g_gan = net_d(real_a, fake_b)
        pred_fake = self.net_d(fake_ab)
        loss_g_gan = self.gan_loss_fn(pred_fake, self.real_label.to(pred_fake).expand_as(pred_fake))

        # 2. real_a -> net_g -> fake_b, real_b-> loss_g_l1
        # loss_g_l1 = ||net_g(real_a) - real_b||
        loss_g_l1 = self.l1_loss_fn(fake_b, real_b) * self.lambda_l1

        loss_g = loss_g_gan + loss_g_l1

        return loss_g


class NetD(nn.Module):
    def __init__(self, in_ch, hidden_ch=64, n_conv=3):
        super().__init__()

        layers = [Conv(in_ch, hidden_ch, k=4, s=2, p=1, bias=True, act=nn.LeakyReLU(0.2), is_norm=False)]

        out_ch = hidden_ch
        for n in range(1, n_conv + 1):
            in_ch = out_ch
            out_ch = hidden_ch * min(2 ** n, 8)

            if n == n_conv:
                layers.append(Conv(in_ch, out_ch, k=4, s=1, p=1, act=nn.LeakyReLU(0.2)))
            else:
                layers.append(Conv(in_ch, out_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2)))

        layers.append(Conv(out_ch, 1, k=4, s=1, p=1, bias=True, is_norm=False, is_act=False))
        self.conv_seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_seq(x)
