import torch
import torch.nn as nn
from ..layers import Conv
from ..semantic_segmentation.Unet import CirUnetBlock, Config as Config_
from utils.torch_utils import ModuleManager
import functools


class Config:
    net_g_config = Config_.unet256

    net_d_config = dict(
        hidden_ch=64,
        n_layers=3,
    )

    @classmethod
    def get(cls, name=None):
        return dict(
            net_g_config=cls.net_g_config,
            net_d_config=cls.net_g_config
        )


class Model(nn.ModuleList):
    """refer to:
    paper:
        - [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
    code:
        - https://github.com/phillipi/pix2pix
        - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """

    def __init__(self, in_ch, net_g_config=Config.net_g_config, net_d_config=Config.net_d_config,
                 net_g=None, net_d=None,
                 lambda_l1=100.0, real_label=1., fake_label=0.,
                 **kwargs):
        super().__init__()

        self.net_g = net_g if net_g is not None else NetG(in_ch, **net_g_config)
        self.net_d = net_d if net_d is not None else NetD(in_ch * 2, **net_d_config)

        ModuleManager.initialize_layers(self.net_g)
        ModuleManager.initialize_layers(self.net_d)

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


class NetG(nn.Sequential):
    def __init__(self, in_ch, **kwargs):
        super().__init__(
            CirUnetBlock(in_ch, in_ch, **kwargs),
            nn.Tanh()
        )


class NetD(nn.Sequential):
    """PatchGAN discriminator"""

    def __init__(self, in_ch, hidden_ch=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        if isinstance(norm_layer, functools.partial):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func is not nn.BatchNorm2d
        else:
            use_bias = norm_layer is not nn.BatchNorm2d

        layers = [Conv(in_ch, hidden_ch, k=4, s=2, p=1, mode='ca', act=nn.LeakyReLU(0.2))]

        out_ch = hidden_ch
        for n in range(1, n_layers + 1):
            in_ch = out_ch
            out_ch = hidden_ch * min(2 ** n, 8)

            if n == n_layers:
                layers.append(Conv(in_ch, out_ch, 4, s=1, p=1, bias=use_bias, mode='cna', act=nn.LeakyReLU(0.2), norm=norm_layer(out_ch)))
            else:
                layers.append(Conv(in_ch, out_ch, 4, s=2, p=1, bias=use_bias, mode='cna', act=nn.LeakyReLU(0.2), norm=norm_layer(out_ch)))

        layers.append(Conv(out_ch, 1, 4, s=1, p=1, mode='c'))
        super().__init__(*layers)
