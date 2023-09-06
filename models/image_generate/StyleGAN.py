import os
import sys
import math
import fire
import json

from tqdm import tqdm
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial
import multiprocessing
from contextlib import contextmanager, ExitStack

import numpy as np

import torch
from torch import nn, einsum
from torch.utils import data
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from einops import rearrange, repeat
from kornia.filters import filter2d

import torchvision
from torchvision import transforms

from vector_quantize_pytorch import VectorQuantize

from PIL import Image
from pathlib import Path

try:
    from apex import amp

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

import aim

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'
from functools import partial
import random
import torch
import torch.nn.functional as F

from ..layers import Linear, Conv, ConvT, EqualLinear, Conv2DMod, Residual


def DiffAugment(x, types=[]):
    for p in types:
        for f in AUGMENT_FNS[p]:
            x = f(x)
    return x.contiguous()


# """
# Augmentation functions got images as `x`
# where `x` is tensor with this dimensions:
# 0 - count of images
# 1 - channels
# 2 - width
# 3 - height of image
# """

def rand_brightness(x, scale):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) * scale
    return x


def rand_saturation(x, scale):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (((torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) * 2.0 * scale) + 1.0) + x_mean
    return x


def rand_contrast(x, scale):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (((torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5) * 2.0 * scale) + 1.0) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_offset(x, ratio=1, ratio_h=1, ratio_v=1):
    w, h = x.size(2), x.size(3)

    imgs = []
    for img in x.unbind(dim=0):
        max_h = int(w * ratio * ratio_h)
        max_v = int(h * ratio * ratio_v)

        value_h = random.randint(0, max_h) * 2 - max_h
        value_v = random.randint(0, max_v) * 2 - max_v

        if abs(value_h) > 0:
            img = torch.roll(img, value_h, 2)

        if abs(value_v) > 0:
            img = torch.roll(img, value_v, 1)

        imgs.append(img)

    return torch.stack(imgs)


def rand_offset_h(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=ratio, ratio_v=0)


def rand_offset_v(x, ratio=1):
    return rand_offset(x, ratio=1, ratio_h=0, ratio_v=ratio)


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'brightness': [partial(rand_brightness, scale=1.)],
    'lightbrightness': [partial(rand_brightness, scale=.65)],
    'contrast': [partial(rand_contrast, scale=.5)],
    'lightcontrast': [partial(rand_contrast, scale=.25)],
    'saturation': [partial(rand_saturation, scale=1.)],
    'lightsaturation': [partial(rand_saturation, scale=.5)],
    'color': [partial(rand_brightness, scale=1.), partial(rand_saturation, scale=1.), partial(rand_contrast, scale=0.5)],
    'lightcolor': [partial(rand_brightness, scale=0.65), partial(rand_saturation, scale=.5), partial(rand_contrast, scale=0.5)],
    'offset': [rand_offset],
    'offset_h': [rand_offset_h],
    'offset_v': [rand_offset_v],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}

# constants

NUM_CORES = multiprocessing.cpu_count()
EXTS = ['jpg', 'jpeg', 'png']


# helper classes


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if not exists(old):
            return new
        return old * self.beta + (1 - self.beta) * new


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random.random() < self.prob else self.fn_else
        return fn(x)


class ChanNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, *_, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)


# helpers

def exists(val):
    return val is not None


# augmentations

def random_hflip(tensor, prob):
    if prob < random.random():
        return tensor
    return torch.flip(tensor, dims=(3,))


class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob=0., types=[], detach=False):
        if random.random() < prob:
            images = random_hflip(images, prob=0.5)
            images = DiffAugment(images, types=types)

        if detach:
            images = images.detach()

        return self.D(images)


# stylegan2 classes
class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul=0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers += [
                EqualLinear(emb, emb, lr_mul),
                nn.LeakyReLU(0.2, inplace=True)
            ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, in_ch, hidden_ch=64, heads=8):
        super().__init__()
        self.scale = hidden_ch ** -0.5
        self.heads = heads
        tmp_ch = hidden_ch * heads

        self.to_q = nn.Conv2d(in_ch, tmp_ch, 1, bias=False)
        self.to_kv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, stride=1, bias=False),
            nn.Conv2d(in_ch, tmp_ch * 2, kernel_size=1, bias=False)
        )

        self.to_out = Conv(tmp_ch, in_ch, 1, act=nn.GELU(), is_norm=False, mode='ac')

    def forward(self, fmap):
        h, x, y = self.heads, *fmap.shape[-2:]
        q, k, v = (self.to_q(fmap), *self.to_kv(fmap).chunk(2, dim=1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h=h), (q, k, v))

        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        q = q * self.scale

        context = einsum('b n d, b n e -> b d e', k, v)
        out = einsum('b n d, b d e -> b n e', q, context)
        out = rearrange(out, '(b h) (x y) d -> b (h d) x y', h=h, x=x, y=y)

        return self.to_out(out)


class AttentionBlock(nn.Sequential):
    """one layer of self-attention and feedforward, for images"""

    def __init__(self, in_ch):
        super().__init__(
            Residual(nn.Sequential(
                ChanNorm(in_ch),
                LinearAttention(in_ch)
            )),
            Residual(nn.Sequential(
                ChanNorm(in_ch),
                Conv(in_ch, in_ch * 2, 1, act=nn.LeakyReLU(0.2, inplace=True), is_norm=False),
                Conv(in_ch * 2, in_ch, 1, is_act=False, is_norm=False),
            ))
        )


class RGBBlock(nn.Module):
    def __init__(self, latent_dim, in_ch, is_upsample, rgba=False):
        super().__init__()
        self.in_channels = in_ch
        self.to_style = nn.Linear(latent_dim, in_ch)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(in_ch, out_filters, 1, demod=False)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            Blur()
        ) if is_upsample else nn.Identity()

    def forward(self, x, prev_rgb, istyle):
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        x = self.upsample(x)
        return x


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, in_ch, filters, is_upsample=True, upsample_rgb=True, rgba=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if is_upsample else nn.Identity()

        self.to_style1 = nn.Linear(latent_dim, in_ch)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(in_ch, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch, out_ch, is_downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(in_ch, out_ch, 1, stride=(2 if is_downsample else 1))

        self.conv_seq = nn.Sequential(
            Conv(in_ch, out_ch, 3, act=nn.LeakyReLU(0.2, inplace=True), is_norm=False),
            Conv(out_ch, out_ch, 3, act=nn.LeakyReLU(0.2, inplace=True), is_norm=False),
        )

        self.downsample = nn.Sequential(
            Blur(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=2)
        ) if is_downsample else nn.Identity()

    def forward(self, x):
        res = self.conv_res(x)
        x = self.conv_seq(x)
        x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity=16, transparent=False, attn_layers=[], no_const=False, fmap_max=512):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        out_ches = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]

        set_fmap_max = partial(min, fmap_max)
        out_ches = list(map(set_fmap_max, out_ches))
        in_ch = out_ches[0]

        # in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(latent_dim, in_ch, 4, 1, 0, bias=False)
        else:
            self.initial_block = nn.Parameter(torch.randn((1, in_ch, 4, 4)))

        self.initial_conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        # for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
        for i, out_ch in enumerate(out_ches):
            not_first = i != 0
            not_last = i != (self.num_layers - 1)
            num_layer = self.num_layers - i

            attn_fn = AttentionBlock(in_ch) if num_layer in attn_layers else None
            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_ch,
                out_ch,
                is_upsample=not_first,
                upsample_rgb=not_last,
                rgba=transparent
            )
            self.blocks.append(block)

            in_ch = out_ch

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        for style, block, attn in zip(styles, self.blocks, self.attns):
            if exists(attn):
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)

        return rgb


class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity=16, fq_layers=[], fq_dict_size=256, attn_layers=[], transparent=False, fmap_max=512):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        in_ch = 3 if not transparent else 4

        out_ches = [(network_capacity * 4) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        out_ches = list(map(set_fmap_max, out_ches))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for i, out_ch in enumerate(out_ches):
            num_layer = i + 1
            is_not_last = i != (len(out_ches) - 1)

            block = DiscriminatorBlock(in_ch, out_ch, is_downsample=is_not_last)
            blocks.append(block)

            # attn_fn = attn_and_ff(out_ch) if num_layer in attn_layers else None
            attn_fn = AttentionBlock(out_ch) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = PermuteToFrom(VectorQuantize(out_ch, fq_dict_size)) if num_layer in fq_layers else None
            quantize_blocks.append(quantize_fn)
            in_ch = out_ch

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = out_ches[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(self.blocks, self.attn_blocks, self.quantize_blocks):
            x = block(x)

            if exists(attn_block):
                x = attn_block(x)

            if exists(q_block):
                x, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze(), quantize_loss


class Model(nn.ModuleList):
    def __init__(self, image_size, latent_dim=512, fmap_max=512, style_depth=8, network_capacity=16, transparent=False,
                 fq_layers=[], fq_dict_size=256, attn_layers=[], no_const=False,
                 lr_mlp=0.1):
        super().__init__()
        # self.ema_updater = EMA(0.995)
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = 6

        self.net_s = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        self.net_g = Generator(image_size, latent_dim, network_capacity, transparent=transparent, attn_layers=attn_layers, no_const=no_const, fmap_max=fmap_max)
        self.net_d = Discriminator(image_size, network_capacity, fq_layers=fq_layers, fq_dict_size=fq_dict_size, attn_layers=attn_layers, transparent=transparent, fmap_max=fmap_max)

        # self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        # self.GE = Generator(image_size, latent_dim, network_capacity, transparent=transparent, attn_layers=attn_layers, no_const=no_const)
        #
        # self.D_cl = None
        #
        # if cl_reg:
        #     from contrastive_learner import ContrastiveLearner
        #     # experimental contrastive loss discriminator regularization
        #     assert not transparent, 'contrastive loss regularization does not work with transparent images yet'
        #     self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer='flatten')

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.net_d, image_size)

        # # turn off grad for exponential moving averages
        # self.SE.requires_grad_(False)
        # self.GE.requires_grad_(False)

        # init weights
        self._init_weights()
        # self.reset_parameter_averaging()

        # # startup apex mixed precision
        # self.fp16 = fp16
        # if fp16:
        #     (self.S, self.G, self.D, self.SE, self.GE), (self.G_opt, self.D_opt) = \
        #         amp.initialize([self.S, self.G, self.D, self.SE, self.GE], [self.G_opt, self.D_opt], opt_level='O1', num_losses=3)

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.net_g.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    # def EMA(self):
    #     def update_moving_average(ma_model, current_model):
    #         for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
    #             old_weight, up_weight = ma_params.data, current_params.data
    #             ma_params.data = self.ema_updater.update_average(old_weight, up_weight)
    #
    #     update_moving_average(self.SE, self.S)
    #     update_moving_average(self.GE, self.G)
    #
    # def reset_parameter_averaging(self):
    #     self.SE.load_state_dict(self.S.state_dict())
    #     self.GE.load_state_dict(self.G.state_dict())

    def loss_d(self, real_x):
        aug_kwargs = {'prob': 0.0, 'types': ['translation', 'cutout']}

        batch_size = real_x.shape[0]

        noise_x = torch.FloatTensor(batch_size, self.image_size, self.image_size, 1).uniform_(0., 1.).to(real_x.device)
        w_styles = self.gen_styles(self.gen_rand_noise_z_list(batch_size, real_x.device))

        fake_x = self.net_g(w_styles, noise_x)
        fake_y, fake_q_loss = self.D_aug(fake_x.clone().detach(), detach=True, **aug_kwargs)

        real_x.requires_grad_(True)
        real_y, real_q_loss = self.D_aug(real_x, **aug_kwargs)

        return (F.relu(1 + real_y) + F.relu(1 - fake_y)).mean()

    def loss_g(self, real_x):
        aug_kwargs = {'prob': 0.0, 'types': ['translation', 'cutout']}
        batch_size = real_x.shape[0]

        noise_x = self.gen_noise_image(batch_size, real_x.device)
        w_styles = self.gen_styles(self.gen_rand_noise_z_list(batch_size, real_x.device))

        fake_x = self.net_g(w_styles, noise_x)
        fake_y, _ = self.D_aug(fake_x, **aug_kwargs)
        return fake_y.mean()

    def gen_rand_noise_z_list(self, batch_size, device):
        tt = int(torch.rand(()).numpy() * self.num_layers)
        return [
            (self.gen_noise_z(batch_size, device), tt),
            (self.gen_noise_z(batch_size, device), self.num_layers - tt),
        ]

    def gen_same_noise_z_list(self, batch_size, device):
        return [(self.gen_noise_z(batch_size, device), self.num_layers)]

    def gen_styles(self, noise_z):
        w = [(self.net_s(z), num_layers) for z, num_layers in noise_z]
        return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in w], dim=1)

    def gen_noise_z(self, batch_size, device):
        return torch.randn(batch_size, self.latent_dim).to(device)

    def gen_noise_image(self, batch_size, device):
        return torch.FloatTensor(batch_size, self.image_size, self.image_size, 1).uniform_(0., 1.).to(device)

    # def styles_def_to_tensor(self, styles_def):
    #     return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)
    #
    # def latent_to_w(self, latent_descr):
    #     return [(self.S(z), num_layers) for z, num_layers in latent_descr]
