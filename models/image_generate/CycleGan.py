import random
import torch
import torch.nn as nn
from .. import Conv, ConvT
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

    def __init__(self, in_ch, input_size, net_g_config=net_g_config, net_d_config=net_d_config,
                 lambda_identity=0.5, real_label=1., fake_label=0.,
                 **kwargs):
        super().__init__()

        self.net_g_a = NetG(in_ch, 3, **net_g_config)
        self.net_g_b = NetG(in_ch, 3, **net_g_config)
        self.net_d_a = NetD(in_ch, **net_d_config)
        self.net_d_b = NetD(in_ch, **net_d_config)

        initialize_layers(self)

        self.gan_loss_fn = nn.MSELoss()
        self.cycle_l1_loss_fn = torch.nn.L1Loss()
        self.idt_l1_loss_fn = torch.nn.L1Loss()

        self.lambda_identity = lambda_identity
        self.real_label = torch.tensor(real_label)
        self.fake_label = torch.tensor(fake_label)

    def loss_g(self, real_x, net_g_x, net_g_y, net_d_x, lambda_x):
        # symmetrically training
        # fake_x is compared with real_x and real_y, rec_x is compared with real_x
        net_d_x.requires_grad_(False)

        if self.lambda_identity > 0:
            # 1. real_x -> net_g_y -> idt_y, real_x -> loss_idt_y
            # loss_idt_y = ||net_g_y(real_x) - real_x|| * lambda_x * lambda_identity
            idt_y = net_g_y(real_x)
            loss_idt_y = self.idt_l1_loss_fn(idt_y, real_x) * lambda_x * self.lambda_identity

        else:
            loss_idt_y = 0

        # 2. real_x -> net_g_x -> fake_y -> net_d_x -> pred_real, real_label -> gan_loss_fn -> loss_g_x
        # loss_g_x = net_d(net_g_x(real_x)) - 1
        fake_y = net_g_x(real_x)
        pred_real = net_d_x(fake_y)
        loss_g_x = self.gan_loss_fn(pred_real, self.real_label.to(pred_real).expand_as(pred_real))

        # 3. real_x -> net_g_x -> fake_y -> net_g_y -> rec_x, real_x -> loss_cycle_x
        # loss_cycle_x = ||net_g_y(net_g_x(real_x)) - real_x|| * lambda_x
        rec_x = net_g_y(fake_y)
        loss_cycle_x = self.cycle_l1_loss_fn(rec_x, real_x) * lambda_x

        return (fake_y, rec_x), (loss_idt_y, loss_g_x, loss_cycle_x)

    def loss_d(self, real_x, fake_x, net_d_y, cacher):
        net_d_y.requires_grad_(True)

        # 1. real_x -> net_d_y -> pred_real, real_label -> loss_d_real
        # loss_d_real = net_d_y(real_x) - 1
        pred_real = net_d_y(real_x)
        loss_d_real = self.gan_loss_fn(pred_real, self.real_label.to(pred_real).expand_as(pred_real))

        cacher.cache_batch(real_x)
        if random.uniform(0, 1) > 0.5:
            fake_x_old = cacher.get_batch(size=len(real_x))
            fake_x_old = torch.stack(fake_x_old)
        else:
            fake_x_old = fake_x

        # 2. fake_x_old -> net_d_y -> pred_fake, fake_label -> loss_d_fake
        # loss_d_fake = net_d_y(fake_x_old) - 0
        pred_fake = net_d_y(fake_x_old.detach())
        loss_d_fake = self.gan_loss_fn(pred_fake, self.fake_label.to(pred_fake).expand_as(pred_fake))

        loss_d_x = (loss_d_real + loss_d_fake) * 0.5

        return loss_d_x


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
