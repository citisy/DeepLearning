import math
from math import floor, log2
import numpy as np
from torch import nn, einsum
from einops import rearrange, repeat
from kornia.filters import filter2d
import torch
import torch.nn.functional as F
from torch.autograd import grad
from ..layers import Linear, Conv, EqualLinear, Residual
from ..losses import HingeGanLoss
from utils.torch_utils import ModuleManager


class Config:
    net_s_config = dict(
        n_layers=8,
        lr_mul=0.1,
    )

    net_g_config = dict(
        network_capacity=16,
        attn_layers=[-1, -2],
        const_input=True
    )

    net_d_config = dict(
        network_capacity=16,
        fq_dict_size=512,
        fq_layers=[-1, -2],
        attn_layers=[-1, -2]
    )

    @classmethod
    def get(cls, name=None):
        return dict(
            net_s_config=cls.net_s_config,
            net_g_config=cls.net_g_config,
            net_d_config=cls.net_d_config
        )


class Model(nn.ModuleList):
    """refer to:
    paper:
        - [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf)
    code:
        - https://github.com/NVlabs/stylegan
        - https://github.com/lucidrains/stylegan2-pytorch

    """

    def __init__(self, img_ch, image_size, net_g_in_ch=512,
                 net_s_config=Config.net_s_config, net_g_config=Config.net_g_config, net_d_config=Config.net_d_config,
                 ):
        super().__init__()
        self.image_size = image_size
        self.net_g_in_ch = net_g_in_ch
        self.num_layers = int(log2(image_size)) - 1

        self.net_s = StyleMap(net_g_in_ch, **net_s_config)
        self.net_g = Generator(net_g_in_ch, out_ch=img_ch, num_layers=self.num_layers, **net_g_config)
        self.net_d = Discriminator(img_ch, num_layers=self.num_layers, **net_d_config)
        self.disc_criterion = HingeGanLoss()

        self.pl_mean = None

        # init weights
        ModuleManager.initialize_layers(self, init_type='kaiming')

    def loss_d(self, real_x, use_gp=False):
        batch_size = real_x.shape[0]

        noise_x = self.gen_noise_image(batch_size, real_x.device)
        # z -> w > net_s -> styles
        if np.random.random() < 0.9:
            styles = self.gen_styles(self.gen_rand_noise_z_list(batch_size, real_x.device))
        else:
            styles = self.gen_styles(self.gen_same_noise_z_list(batch_size, real_x.device))

        # styles + noise_x -> net_g -> fake_x
        fake_x = self.net_g(styles, noise_x)
        fake_y, fake_q_loss = self.net_d(fake_x.clone().detach())

        real_x.requires_grad_(True)
        real_y, real_q_loss = self.net_d(real_x)

        loss = self.disc_criterion(real_y, fake_y)

        if use_gp:
            # gradient penalty
            loss_gp = gradient_penalty(real_x, real_y)
            loss = loss + loss_gp

        return loss

    def loss_g(self, real_x, use_pp=False, beta=0.99):
        batch_size = real_x.shape[0]

        noise_x = self.gen_noise_image(batch_size, real_x.device)
        if np.random.random() < 0.9:
            styles = self.gen_styles(self.gen_rand_noise_z_list(batch_size, real_x.device))
        else:
            styles = self.gen_styles(self.gen_same_noise_z_list(batch_size, real_x.device))

        fake_x = self.net_g(styles, noise_x)
        fake_y, _ = self.net_d(fake_x)
        loss = fake_y.mean()

        if use_pp:
            # path penalty
            pl_lengths = calc_pl_lengths(styles, fake_x)
            avg_pl_length = torch.mean(pl_lengths.detach())

            if self.pl_mean is not None:
                pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                loss = loss + pl_loss
                self.pl_mean = self.pl_mean * beta + (1 - beta) * avg_pl_length
            else:
                self.pl_mean = avg_pl_length

        return loss

    def gen_rand_noise_z_list(self, batch_size, device):
        tt = int(torch.rand(()).numpy() * self.num_layers)
        return [
            (self.gen_noise_z(batch_size, device), tt),
            (self.gen_noise_z(batch_size, device), self.num_layers - tt),
        ]

    def gen_same_noise_z_list(self, batch_size, device):
        return [(self.gen_noise_z(batch_size, device), self.num_layers)]

    def gen_styles(self, noise_z):
        styles = []
        for z, n in noise_z:
            style = self.net_s(z)
            style = style[:, None, :].expand(-1, n, -1)
            styles.append(style)
        return torch.cat(styles, dim=1)

    def gen_noise_z(self, batch_size, device):
        return torch.randn(batch_size, self.net_g_in_ch).to(device)

    def gen_noise_image(self, batch_size, device):
        return torch.FloatTensor(batch_size, self.image_size, self.image_size, 1).uniform_(0., 1.).to(device)


class StyleMap(nn.Module):
    def __init__(self, in_features, n_layers, lr_mul=0.1):
        super().__init__()
        self.in_features = in_features

        layers = []
        for i in range(n_layers):
            layers.append(Linear(
                in_features, in_features,
                linear=EqualLinear,
                lr_mul=lr_mul,
                act=nn.LeakyReLU(0.2),
                is_norm=False
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, in_ch, out_ch=3, num_layers=6, network_capacity=16, attn_layers=(), const_input=True):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch

        attn_layers = [i % num_layers for i in attn_layers]
        out_ches = [network_capacity * (2 ** (i + 1)) for i in range(num_layers)][::-1]
        out_ches = [min(network_capacity * 32, out_ch) for out_ch in out_ches]  # max=512
        out_ch = out_ches[0]

        self.const_input = const_input

        if const_input:
            self.initial_block = nn.Parameter(torch.randn((1, out_ch, 4, 4)))
        else:
            self.initial_block = nn.ConvTranspose2d(in_ch, out_ch, 4, 1, 0, bias=False)

        in_ch = out_ch
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)

        attention_blocks = []
        blocks = []
        for i, out_ch in enumerate(out_ches):
            attention_blocks.append(AttentionBlock(in_ch) if i in attn_layers else nn.Identity())
            blocks.append(SynthesisBlock(
                self.in_channels,
                in_ch,
                out_ch,
                self.out_channels,
                is_first=i == 0,
                is_last=i == (num_layers - 1),
            ))

            in_ch = out_ch

        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, styles, noise):
        batch_size = styles.shape[0]

        if self.const_input:
            x = self.initial_block.expand(batch_size, -1, -1, -1)
        else:
            # use style to initial the input
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.initial_block(avg_style)

        img = None
        styles = styles.transpose(0, 1)  # (b, n, c) > (n, b, c)
        x = self.conv(x)

        for style, block, attn in zip(styles, self.blocks, self.attention_blocks):
            x = attn(x)
            x, img = block(x, img, style, noise)

        return img


class SynthesisBlock(nn.Module):
    def __init__(self, in_features, in_ch, hidden_ch, out_ch=3, is_first=True, is_last=True):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if not is_first else nn.Identity()

        self.to_style1 = nn.Linear(in_features, in_ch)
        self.combine1 = Conv2DMod(in_ch, hidden_ch, 3)
        self.to_noise1 = nn.Linear(1, hidden_ch)

        self.to_style2 = nn.Linear(in_features, hidden_ch)
        self.combine2 = Conv2DMod(hidden_ch, hidden_ch, 3)
        self.to_noise2 = nn.Linear(1, hidden_ch)

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.head = Head(in_features, hidden_ch, not is_last, out_ch)

    def initialize_layers(self):
        ModuleManager.initialize_layers(self, init_type='kaiming')
        nn.init.zeros_(self.to_noise1.weight)
        nn.init.zeros_(self.to_noise2.weight)
        nn.init.zeros_(self.to_noise1.bias)
        nn.init.zeros_(self.to_noise2.bias)

    def forward(self, x, prev_img, style, noise):
        x = self.upsample(x)
        noise = noise[:, :x.shape[2], :x.shape[3], :]

        style1 = self.to_style1(style)
        x = self.combine1(x, style1)

        noise1 = self.to_noise1(noise).permute((0, 3, 2, 1))
        x = self.act(x + noise1)

        style2 = self.to_style2(style)
        x = self.combine2(x, style2)

        noise2 = self.to_noise2(noise).permute((0, 3, 2, 1))
        x = self.act(x + noise2)

        img = self.head(x, prev_img, style)
        return x, img


class Head(nn.Module):
    def __init__(self, in_features, in_ch, is_upsample, out_ch=3):
        super().__init__()
        self.in_features = in_features
        self.in_channels = in_ch

        # self.to_style = EqualLinear(in_features, in_ch)
        # self.combine = nn.Sequential(
        #     AdaIN(in_ch),
        #     nn.Conv2d(in_ch, out_ch, 1)
        # )

        self.to_style = nn.Linear(in_features, in_ch)
        self.combine = Conv2DMod(in_ch, out_ch, 1, demod=False)
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Blur()
        ) if is_upsample else nn.Identity()

    def forward(self, x, prev_img, style):
        style = self.to_style(style)
        img = self.combine(x, style)

        if prev_img is not None:
            img = img + prev_img

        img = self.upsample(img)
        return img


class AdaIN(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.in_channels = in_ch
        self.norm = nn.InstanceNorm2d(in_ch)

    def forward(self, x, y):
        y = y.unsqueeze(2).unsqueeze(3)
        gamma, beta = y.chunk(2, 1)

        x = self.norm(x)
        x = gamma * x + beta

        return x


class Conv2DMod(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1, dilation=1, demod=True, eps=1e-8, **kwargs):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.k = k
        self.stride = stride
        self.dilation = dilation
        self.demod = demod
        self.weight = nn.Parameter(torch.randn((out_ch, in_ch, k, k)))
        self.eps = eps

    def initialize_layers(self):
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size):
        return ((size - 1) * (self.stride - 1) + self.dilation * (self.k - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        y = y[:, None, :, None, None]
        weight = self.weight[None, :, :, :, :]
        weights = weight * (y + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_channels, *ws)

        padding = self._get_same_padding(h)
        x = F.conv2d(x, weights, padding=padding, groups=b)
        x = x.reshape(-1, self.out_channels, h, w)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_ch, num_layers=6, network_capacity=16, fq_layers=(), fq_dict_size=256, attn_layers=()):
        super().__init__()
        self.in_channels = in_ch

        attn_layers = [i % num_layers for i in attn_layers]
        fq_layers = [i % num_layers for i in fq_layers]
        out_ches = [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]
        out_ches = [min(network_capacity * 32, out_ch) for out_ch in out_ches]  # max=512

        blocks = []
        quantize_blocks = []
        for i, out_ch in enumerate(out_ches):
            blocks.append(nn.Sequential(
                DiscriminatorBlock(in_ch, out_ch, is_last=i == (len(out_ches) - 1)),
                AttentionBlock(out_ch) if i in attn_layers else nn.Identity()
            ))

            quantize_fn = None
            if i in fq_layers:
                from vector_quantize_pytorch import VectorQuantize
                quantize_fn = VectorQuantize(out_ch, fq_dict_size, accept_image_fmap=True)
            quantize_blocks.append(quantize_fn)

            in_ch = out_ch

        self.blocks = nn.ModuleList(blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        out_features = 1

        self.out = nn.Sequential(
            Conv(in_ch, in_ch, 3, p=1, mode='ca', act=nn.LeakyReLU(0.2)),
            nn.Flatten(),
            nn.LazyLinear(out_features)
        )
        self.out_features = out_features

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, q_block) in zip(self.blocks, self.quantize_blocks):
            x = block(x)

            if q_block is not None:
                x, *_, loss = q_block(x)
                quantize_loss += loss

        x = self.out(x)
        return x.squeeze(), quantize_loss


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_last=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, stride=(2 if not is_last else 1))
        self.conv_seq = nn.Sequential(
            Conv(in_ch, out_ch, 3, mode='ca', act=nn.LeakyReLU(0.2)),
            Conv(out_ch, out_ch, 3, mode='ca', act=nn.LeakyReLU(0.2)),
        )

        self.down_sample = nn.Sequential(
            Blur(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=2)
        ) if not is_last else nn.Identity()

    def forward(self, x):
        y = self.conv(x)
        x = self.conv_seq(x)
        x = self.down_sample(x)
        x = (x + y) * (1 / math.sqrt(2))
        return x


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.Tensor([1, 2, 1]))

    def forward(self, x):
        w = self.weight
        w = w[None, None, :] * w[None, :, None]
        return filter2d(x, w, normalized=True)


class AttentionBlock(nn.Sequential):
    """one layer of self-attention and feedforward, for images"""

    def __init__(self, in_ch):
        super().__init__(
            Residual(nn.Sequential(
                ChannelNorm(in_ch),
                ConvAttention(in_ch),
            ), is_norm=False),
            Residual(nn.Sequential(
                Conv(in_ch, in_ch * 2, 1, mode='nca', act=nn.LeakyReLU(0.2, inplace=True), norm=ChannelNorm(in_ch)),
                Conv(in_ch * 2, in_ch, 1, mode='c'),
            ), is_norm=False)
        )


class ChannelNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class ConvAttention(nn.Module):
    def __init__(self, in_ch, hidden_ch=64, heads=8):
        super().__init__()
        self.scale = hidden_ch ** -0.5
        self.heads = heads
        tmp_ch = hidden_ch * heads

        self.to_q = nn.Conv2d(in_ch, tmp_ch, 1, bias=False)
        self.to_kv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, stride=1, bias=False),
            nn.Conv2d(in_ch, tmp_ch * 2, 1, bias=False)
        )

        self.to_out = Conv(tmp_ch, in_ch, 1, mode='ac', act=nn.GELU())

    def forward(self, fmap):
        x, y = fmap.shape[-2:]
        q = self.to_q(fmap)
        k, v = self.to_kv(fmap).chunk(2, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=self.heads), (q, k, v))

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=self.heads, x=x, y=y)

        return self.to_out(out)


def gradient_penalty(inputs, outputs, weight=10):
    batch_size = inputs.shape[0]
    gradients = grad(outputs=outputs, inputs=inputs,
                     grad_outputs=torch.ones(outputs.size(), device=inputs.device),
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.reshape(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def calc_pl_lengths(styles, images):
    device = images.device
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = grad(outputs=outputs, inputs=styles,
                    grad_outputs=torch.ones(outputs.shape, device=device),
                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()
