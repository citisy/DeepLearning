import math
from math import floor, log2
import numpy as np
from random import random
from torch import nn, einsum
from einops import rearrange, repeat, reduce
import torch
import torch.nn.functional as F
from torch.autograd import grad
from ..layers import Linear, Conv, EqualLinear, Residual
from ..semantic_segmentation.Unet import CurBlock
from utils.torch_utils import initialize_layers
from collections import namedtuple
from functools import partial
from einops.layers.torch import Rearrange
from .ddpm_ import Unet
from tqdm import tqdm


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


PRED_X0 = 1
PRED_Z = 2
PRED_V = 3


class Model(nn.ModuleList):
    def __init__(self, img_ch, image_size, timesteps=300, offset_noise_strength=0., objective=PRED_V,
                 min_snr_loss_weight=False, min_snr_gamma=5
                 ):
        super().__init__()
        self.image_size = image_size
        self.img_ch = img_ch
        self.timesteps = timesteps
        self.offset_noise_strength = offset_noise_strength

        # pred_z -> model(x_t, t) = z_t
        # pred_x0 -> model(x_t, t) = x_0
        # pred_v -> model(x_t, t) = v_t = z_t \sqrt ca_t - x_0 * \sqrt{1-ca_t}
        self.objective = objective

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # define beta schedule
        betas = linear_beta_schedule(timesteps=timesteps)  # (timesteps, )

        # define alphas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # # ca_t = \cumprod a_t
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)  # ca_{t-1} = \cumprod a_{t-1}

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # derive loss weight
        # snr - signal noise ratio
        snr = alphas_cumprod / (1 - alphas_cumprod)

        # https://arxiv.org/abs/2303.09556
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == PRED_Z:
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif objective == PRED_X0:
            register_buffer('loss_weight', maybe_clipped_snr)
        elif objective == PRED_V:
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        self.model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            flash_attn=False
        )
        self.self_condition = self.model.self_condition

    def forward(self, x):
        if self.training:
            b, c, h, w = x.shape
            t = torch.randint(0, self.timesteps, (b,), device=x.device).long()
            return {'loss': self.loss(x, t)}
        else:
            return self.post_process(x)

    def model_predictions(self, x_t, t, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False):
        """x_t, t -> pred_z_t, x_0"""
        model_output = self.model(x_t, t, x_self_cond)

        if self.objective == PRED_Z:
            pred_noise = model_output
            x_0 = self.predict_start_from_noise(x_t, t, pred_noise)
            if clip_x_start:
                x_0 = torch.clamp(x_0, min=-1., max=1.)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x_t, t, x_0)

        elif self.objective == PRED_X0:
            x_0 = model_output
            if clip_x_start:
                x_0 = torch.clamp(x_0, min=-1., max=1.)
            pred_noise = self.predict_noise_from_start(x_t, t, x_0)

        elif self.objective == PRED_V:
            v = model_output
            x_0 = self.predict_start_from_v(x_t, t, v)
            if clip_x_start:
                x_0 = torch.clamp(x_0, min=-1., max=1.)
            pred_noise = self.predict_noise_from_start(x_t, t, x_0)

        else:
            raise ValueError(f'unknown objective {self.objective}')

        return pred_noise, x_0

    def predict_start_from_v(self, x_t, t, v):
        # x_0 = x_t * \sqrt ca_t - v_t * \sqrt{1-ca_t}
        # where, v_t = z_t \sqrt ca_t - x_0 * \sqrt{1-ca_t}
        # get, x_0 = (x_t - z_t * \sqrt{1-ca_t}) / \sqrt ca_t
        return extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v

    def predict_noise_from_start(self, x_t, t, x_0):
        # z_t = (x_t * \sqrt{1/ca_t} - x_0) / \sqrt{1/ca_t - 1}
        # come from x_0 = 1/ca_t * (x_t - z_t * \sqrt{1-ca_t}
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_0) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def predict_start_from_noise(self, x_t, t, noise):
        # x_0 = x_t * \sqrt{1/ca_t} - z_t * \sqrt{1/ca_t - 1}
        # come from x_0 = 1/ca_t * (x_t - z_t * \sqrt{1-ca_t}
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def predict_v(self, x_0, t, noise):
        # v_t = z_t \sqrt ca_t - x_0 * \sqrt{1-ca_t}
        return extract(self.sqrt_alphas_cumprod, t, x_0.shape) * noise - extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * x_0

    def loss(self, x_0, t, noise=None, offset_noise_strength=None):
        x_0 = x_0 * 2 - 1  # normalize
        noise = default(noise, lambda: torch.randn_like(x_0))

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_0.shape[:2], device=self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample
        x_t = self.q_sample(x_0, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                _, x_self_cond = self.model_predictions(x_t, t)
                x_self_cond.detach_()

        pred = self.model(x_t, t, x_self_cond)

        if self.objective == PRED_Z:
            real = noise
        elif self.objective == PRED_X0:
            real = x_0
        elif self.objective == PRED_V:
            real = self.predict_v(x_0, t, noise)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(pred, real, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def q_sample(self, x0, t, noise=None):
        # x_t = x_0 * \sqrt ca_t + z_t * \sqrt (1 - ca_t)
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise

    def gen_x_t(self, batch_size, device):
        return torch.randn((batch_size, 3, self.image_size, self.image_size), device=device)

    def post_process(self, x_t, return_all_timesteps=False):
        imgs = [x_t]
        x_0 = None

        # t: T-1 -> 0
        for t in reversed(range(0, self.timesteps)):
            self_cond = x_0 if self.self_condition else None
            x_t, x_0 = self.p_sample(x_t, t, self_cond)
            imgs.append(x_t)

        ret = x_t if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = (ret + 1) * 0.5  # unnormalize
        return ret

    @torch.inference_mode()
    def p_sample(self, x_t, t: int, x_self_cond=None):
        batched_times = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_0 = self.p_mean_variance(x_t, t=batched_times, x_self_cond=x_self_cond, clip_denoised=True)
        noise = torch.randn_like(x_t) if t > 0 else 0.  # no noise if t == 0

        # x_{t-1} = u(x_t, x_0) + exp(0.5 * 2 * \log s(x_t, t)) * z_t
        # where, exp((0.5 * 2 * \log s(x_t, t))) = s(x_t, t), z~N(0, 1)
        x_t_1 = model_mean + (0.5 * model_log_variance).exp() * noise
        return x_t_1, x_0

    def p_mean_variance(self, x_t, t, x_self_cond=None, clip_denoised=True):
        _, x_0 = self.model_predictions(x_t, t, x_self_cond)

        if clip_denoised:
            x_0.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_0, x_t, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_0

    def q_posterior(self, x_0, x_t, t):
        # u(x_t, x_0) = (b * \sqrt ca_{t-1} / (1-ca_t)) * x0 + ((1-ca_{t-1}) * \sqrt a_t / (1 - ca_t)) * x_t
        posterior_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t

        # s(x_t, t)^2 = b * (1-ca_{t-1}) / (1-ca_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)

        # \log s(x_t, t)^2 = 2 \log s(x_t, t)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped


def extract(a, t, x_shape):
    """return a_t with shape of (b, 1, 1, ...)"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Backbone(nn.Module):
    pass


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class LinearAttention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            num_mem_kv=4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            num_mem_kv=4,
            flash=False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class Attend(nn.Module):
    def __init__(
            self,
            dropout=0.,
            flash=False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        AttentionConfig = namedtuple('AttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])
        # determine efficient attention configs for cuda and cpu

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = AttentionConfig(True, False, False)
        else:
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention
        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash:
            return self.flash_attn(q, k, v)

        scale = q.shape[-1] ** -0.5

        # similarity
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )
