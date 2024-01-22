import math
from random import random
from torch import nn, einsum
from einops import rearrange, repeat, reduce
import torch
import torch.nn.functional as F
from ..layers import Linear, Conv, SelfAttention3D, LinearSelfAttention3D
from functools import partial
from einops.layers.torch import Rearrange

PRED_X0 = 1
PRED_Z = 2
PRED_V = 3


class Config:
    in_module_config = dict(self_condition=False)
    backbone_config = dict(
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        resnet_block_groups=8,
        dim_factors=(1, 2, 4, 8),
        attn_heads=4,
        attn_dim_heads=32
    )

    @classmethod
    def get(cls, name=None):
        return dict(
            in_module_config=cls.in_module_config,
            backbone_config=cls.backbone_config
        )


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


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1):
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


class Model(nn.ModuleList):
    """refer to:
    paper:
        - Denoising Diffusion Probabilistic Models
    code:
        - https://github.com/hojonathanho/diffusion
        - https://github.com/lucidrains/denoising-diffusion-pytorch
    """

    def __init__(self, img_ch, image_size,
                 schedule_func=linear_beta_schedule, timesteps=300,
                 offset_noise_strength=0., objective=PRED_V,
                 min_snr_loss_weight=False, min_snr_gamma=5, model_kwargs=Config.get()
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

        self.register_schedule(schedule_func, timesteps, min_snr_loss_weight, min_snr_gamma)
        self.model = Diffusion(**model_kwargs)
        self.self_condition = False

    def register_schedule(self, schedule_func, timesteps, min_snr_loss_weight, min_snr_gamma):
        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # define beta schedule
        betas = schedule_func(timesteps=timesteps)  # (timesteps, )

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

        if self.objective == PRED_Z:
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif self.objective == PRED_X0:
            register_buffer('loss_weight', maybe_clipped_snr)
        elif self.objective == PRED_V:
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

    def forward(self, x):
        if self.training:
            b, c, h, w = x.shape
            t = torch.randint(0, self.timesteps, (b,), device=x.device).long()
            return {'loss': self.loss(x, t)}
        else:
            return self.post_process(x)

    def model_predictions(self, x_t, t, x_self_cond=None, clip_x_start=False,
                          return_pred_noise=False, rederive_pred_noise=False):
        """x_t, t -> pred_z_t, x_0"""
        model_output = self.model(x_t, t, x_self_cond)

        pred_noise = None
        if self.objective == PRED_Z:
            pred_noise = model_output
            x_0 = self.predict_start_from_noise(x_t, t, pred_noise)
            if clip_x_start:
                x_0 = torch.clamp(x_0, min=-1., max=1.)

            if return_pred_noise and clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x_t, t, x_0)

        elif self.objective == PRED_X0:
            x_0 = model_output
            if clip_x_start:
                x_0 = torch.clamp(x_0, min=-1., max=1.)
            if return_pred_noise:
                pred_noise = self.predict_noise_from_start(x_t, t, x_0)

        elif self.objective == PRED_V:
            v = model_output
            x_0 = self.predict_start_from_v(x_t, t, v)
            if clip_x_start:
                x_0 = torch.clamp(x_0, min=-1., max=1.)
            if return_pred_noise:
                pred_noise = self.predict_noise_from_start(x_t, t, x_0)

        else:
            raise ValueError(f'unknown objective {self.objective}')

        return x_0, pred_noise

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
        x_0 = x_0 * 2 - 1  # normalize, [0, 1] -> [-1, 1]
        if noise is None:
            noise = torch.randn_like(x_0)

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        if offset_noise_strength is None:
            offset_noise_strength = self.offset_noise_strength

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
                x_self_cond, _ = self.model_predictions(x_t, t)
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
        images = [x_t]
        x_0 = None

        # t: T-1 -> 0
        for t in reversed(range(0, self.timesteps)):
            self_cond = x_0 if self.self_condition else None
            x_t, x_0 = self.p_sample(x_t, t, self_cond)
            images.append(x_t)

        images = x_t if not return_all_timesteps else torch.stack(images, dim=1)
        images = (images + 1) * 0.5  # unnormalize, [-1, 1] -> [0, 1]
        return images

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
        x_0, _ = self.model_predictions(x_t, t, x_self_cond)

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


class Diffusion(nn.Module):
    def __init__(self, in_ch=3, hidden_ch=64,
                 in_module_config=dict(), backbone_config=dict(),
                 ):
        super().__init__()

        self.in_module = InModule(in_ch, hidden_ch, **in_module_config)
        self.backbone = Backbone(hidden_ch, **backbone_config)
        self.out_module = nn.Conv2d(hidden_ch, in_ch, 1)

    def forward(self, x, time, x_self_cond=None):
        x = self.in_module(x, x_self_cond)
        x = self.backbone(x, time)
        x = self.out_module(x)
        return x


class InModule(nn.Module):
    def __init__(self, in_ch, out_ch, self_condition=False):
        super().__init__()

        self.self_condition = self_condition
        in_ch = in_ch * (2 if self_condition else 1)
        self.conv = Conv(in_ch, out_ch, 7, mode='c')

    def forward(self, x, x_self_cond=None):
        if self.self_condition:
            if x_self_cond is None:
                x_self_cond = torch.zeros_like(x)
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.conv(x)
        return x


class Backbone(nn.Module):
    """base on Unet, add attention, res, etc."""

    def __init__(self, dim,
                 learned_sinusoidal_cond=False,
                 random_fourier_features=False,
                 learned_sinusoidal_dim=16,
                 sinusoidal_pos_emb_theta=10000,
                 resnet_block_groups=8,
                 dim_factors=(1, 2, 4, 8),
                 attn_heads=4,
                 attn_dim_heads=32
                 ):
        super().__init__()

        if learned_sinusoidal_cond:
            sin_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sin_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            sin_pos_emb,
            Linear(fourier_dim, time_dim, mode='la', act=nn.GELU()),
            Linear(time_dim, time_dim, mode='l'),
        )

        num_stages = len(dim_factors)

        if isinstance(attn_heads, int):
            attn_heads = [attn_heads] * num_stages

        if isinstance(attn_dim_heads, int):
            attn_dim_heads = [attn_dim_heads] * num_stages

        res_block = partial(ResnetBlock, groups=resnet_block_groups, time_emb_dim=time_dim)

        in_ch = dim
        self.downs = nn.ModuleList([])
        for i in range(num_stages):
            is_bottom = i == num_stages - 1
            attn_layer = SelfAttention3D if is_bottom else partial(LinearSelfAttention3D, norm=RMSNorm(in_ch))
            out_ch = dim * dim_factors[i]

            self.downs.append(nn.ModuleList([
                res_block(in_ch, in_ch),
                res_block(in_ch, in_ch),
                RMSNorm(in_ch),
                attn_layer(in_ch, head_dim=attn_dim_heads[i], n_heads=attn_heads[i]),
                nn.Conv2d(in_ch, out_ch, 3, padding=1) if is_bottom else nn.Sequential(
                    Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
                    nn.Conv2d(in_ch * 4, out_ch, 1)
                ),
            ]))

            in_ch = out_ch

        out_ch = in_ch
        self.mid_block1 = res_block(out_ch, out_ch)
        self.mid_attn = SelfAttention3D(out_ch, n_heads=attn_heads[-1], head_dim=attn_dim_heads[-1])
        self.mid_block2 = res_block(out_ch, out_ch)

        self.ups = nn.ModuleList([])
        for i in reversed(range(num_stages)):
            is_top = i == 0
            is_bottom = i == num_stages - 1
            attn_layer = SelfAttention3D if is_bottom else partial(LinearSelfAttention3D, norm=RMSNorm(out_ch))
            in_ch = dim if i == 0 else dim * dim_factors[i - 1]

            self.ups.append(nn.ModuleList([
                res_block(out_ch + in_ch, out_ch),
                res_block(out_ch + in_ch, out_ch),
                RMSNorm(out_ch),
                attn_layer(out_ch, head_dim=attn_dim_heads[i], n_heads=attn_heads[i]),
                nn.Conv2d(out_ch, in_ch, 3, padding=1) if is_top else nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(out_ch, in_ch, 3, padding=1)
                ),
            ]))

            out_ch = in_ch

        self.final_res_block = res_block(in_ch * 2, in_ch)

    def forward(self, x, time):
        r = x.clone()
        t = self.time_mlp(time)
        h = []

        for block1, block2, norm, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = norm(x)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, norm, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = norm(x)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return x


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """following @crowsonkb's lead with random (learned optional) sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim).unsqueeze(0), requires_grad=not is_random)

    def forward(self, x):
        x = x[:, None]
        freqs = x * self.weights * 2 * torch.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        half_dim = dim // 2
        log_theta = math.log(theta)
        emb = log_theta / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)

    def forward(self, x):
        emb = self.emb
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = Linear(time_emb_dim, out_ch * 2, mode='al', act=nn.SiLU()) if time_emb_dim else None

        self.block1 = ResBlock(in_ch, out_ch, groups=groups)
        self.block2 = ResBlock(out_ch, out_ch, groups=groups)
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb[:, :, None, None]
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.proj(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, groups=8):
        super().__init__()
        self.proj = Conv(in_ch, out_ch, 3, p=1, mode='cn', norm=nn.GroupNorm(groups, out_ch))
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)
