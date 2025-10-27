import math
from functools import partial
from random import random

import torch
import torch.nn.functional as F
from einops import rearrange, reduce
from torch import nn

from utils import torch_utils
from .. import bundles, attentions, activations, normalizations
from ..layers import Linear, Conv, Upsample, Downsample
from ..embeddings import SinusoidalEmbedding


class Config(bundles.Config):
    PRED_X0 = 1
    PRED_Z = 2
    PRED_V = 3

    LINEAR = 1
    COSINE = 2
    SIGMOID = 3

    sampler = dict(
        objective=PRED_V,
        schedule_type=LINEAR
    )

    in_module = dict(self_condition=False)

    backbone = dict(
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
    def make_full_config(cls) -> dict:
        return {
            '': dict(
                sampler_config=cls.sampler,
                in_module_config=cls.in_module,
                backbone_config=cls.backbone
            )
        }


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


def extract(a, t, x_shape):
    """return a_t with shape of (b, 1, 1, ...)"""
    out = a.gather(-1, t)
    return append_dims(out, len(x_shape))


def linear_beta_schedule(timesteps, start=None, end=None):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    start = start or scale * 0.0001
    end = end or scale * 0.02
    return torch.linspace(start, end, timesteps, dtype=torch.float64)


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


def make_beta_schedule(name=Config.LINEAR):
    d = {
        Config.LINEAR: linear_beta_schedule,
        Config.COSINE: cosine_beta_schedule,
        Config.SIGMOID: sigmoid_beta_schedule
    }
    return d.get(name)


make_norm_fn = partial(normalizations.GroupNorm32, eps=1e-5, affine=True)


class Model(nn.ModuleList):
    """refer to:
    paper:
        - Denoising Diffusion Probabilistic Models
    code:
        - https://github.com/hojonathanho/diffusion
        - https://github.com/lucidrains/denoising-diffusion-pytorch
    """

    use_half = True
    low_memory_run = True

    def __init__(self, img_ch, image_size, **configs):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = image_size
        self.img_ch = img_ch
        self.__dict__.update(configs.get('model_config', {}))

        self.make_sampler(**configs)
        self.make_diffuse(**configs)

    _device = None
    _dtype = None

    @property
    def device(self):
        return torch_utils.ModuleInfo.possible_device(self) if self._device is None else self._device

    @property
    def dtype(self):
        return torch_utils.ModuleInfo.possible_dtype(self) if self._dtype is None else self._dtype

    def set_low_memory_run(self):
        # Not critical to run single batch for decoding strategy, but reduce more GPU memory
        self.vae.encode = partial(torch_utils.ModuleManager.single_batch_run, self.vae, self.vae.encode)
        self.vae.decode = partial(torch_utils.ModuleManager.single_batch_run, self.vae, self.vae.decode)

        def wrap1(module, func):
            # note, device would be changed after model initialization.
            def wrap2(*args, **kwargs):
                return torch_utils.ModuleManager.low_memory_run(module, func, self.device, *args, **kwargs)

            return wrap2

        self.make_txt_cond = wrap1(self.cond, self.make_txt_cond)
        self.sampler.forward = wrap1(self.backbone, self.sampler.forward)
        self.sampler.loss = wrap1(self.backbone, self.sampler.loss)
        self.vae.encode = wrap1(self.vae, self.vae.encode)
        self.vae.decode = wrap1(self.vae, self.vae.decode)
        self.sampler.to(self.device)    # for init meta data

    def set_half(self):
        # note, vanilla sdxl vae can not convert to fp16, but bf16
        # or use a special vae checkpoint, e.g. https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
        dtype = torch.bfloat16

        torch_utils.ModuleManager.apply(
            self,
            lambda module: module.to(dtype),
            include=['cond', 'backbone', 'vae'],
            exclude=[normalizations.GroupNorm32]
        )

        modules = [self.sampler]
        if hasattr(self.sampler, 'schedule'):
            modules.append(self.sampler.schedule)

        for module in modules:
            for name, tensor in module.named_buffers():
                setattr(module, name, tensor.to(dtype))

        self.make_txt_cond = partial(torch_utils.ModuleManager.assign_dtype_run, self.cond, self.make_txt_cond, dtype, force_effect_module=False)
        self.sampler.forward = partial(torch_utils.ModuleManager.assign_dtype_run, self.backbone, self.sampler.forward, dtype, force_effect_module=False)
        self.sampler.loss = partial(torch_utils.ModuleManager.assign_dtype_run, self.backbone, self.sampler.loss, dtype, force_effect_module=False)
        self.vae.encode = partial(torch_utils.ModuleManager.assign_dtype_run, self.vae, self.vae.encode, dtype, force_effect_module=False)
        self.vae.decode = partial(torch_utils.ModuleManager.assign_dtype_run, self.vae, self.vae.decode, dtype, force_effect_module=False)

    def make_sampler(self, sampler_config=Config.sampler, **kwargs):
        self.sampler = Sampler(**sampler_config)

    def make_diffuse(self, in_module_config=Config.in_module, backbone_config=Config.backbone, **kwargs):
        self.in_module = InModule(self.img_ch, self.hidden_ch, **in_module_config)
        self.backbone = UNetModel(self.hidden_ch, **backbone_config)
        self.head = nn.Conv2d(self.hidden_ch, self.img_ch, 1)

    def diffuse(self, x, time, x_self_cond=None, **backbone_kwargs):
        x = self.in_module(x, x_self_cond)
        x = self.backbone(x, time, **backbone_kwargs)
        x = self.head(x)
        return x

    def forward(self, *args, **kwargs):
        if self.training:
            return self.fit(*args, **kwargs)
        else:
            return self.inference(*args, **kwargs)

    def fit(self, x, **kwargs):
        # if training, x is x0, the real image
        return {'loss': self.sampler.loss(self.diffuse, x, **kwargs)}

    def inference(self, x, **kwargs):
        # if inference, x is x_T, the noise
        return self.sampler(self.diffuse, x, **kwargs)

    @property
    def diffuse_in_ch(self):
        """channels of x_t"""
        return self.img_ch

    def diffuse_in_size(self, image_size=None):
        """size of x_t"""
        image_size = image_size or self.image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        return image_size

    def gen_x_t(self, batch_size, image_size=None):
        return torch.randn((batch_size, self.diffuse_in_ch, *self.diffuse_in_size(image_size)[::-1]), device=self.device, dtype=self.dtype)


class Sampler(nn.Module):
    schedule_type = Config.LINEAR
    timesteps = 1000

    # pred_z -> model(x_t, t) = z_t
    # pred_x0 -> model(x_t, t) = x_0
    # pred_v -> model(x_t, t) = v_t = z_t \sqrt ca_t - x_0 * \sqrt{1-ca_t}
    objective = Config.PRED_V
    self_condition = False

    min_snr_loss_weight = False
    min_snr_gamma = 5

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.make_schedule()

    def make_schedule(self):
        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32), persistent=False)

        # define beta schedule
        betas = make_beta_schedule(self.schedule_type)(timesteps=self.timesteps)  # (timesteps, )

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
        if self.min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=self.min_snr_gamma)

        if self.objective == Config.PRED_Z:
            register_buffer('loss_weight', maybe_clipped_snr / snr)
        elif self.objective == Config.PRED_X0:
            register_buffer('loss_weight', maybe_clipped_snr)
        elif self.objective == Config.PRED_V:
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

    def _apply(self, fn, recurse=True):
        """apply for meta load"""
        if self.betas.is_meta:
            self.make_schedule()
        return super()._apply(fn, recurse)

    @property
    def num_steps(self):
        return self.timesteps

    def make_timesteps(self, i0=None):
        timestep_seq = range(0, self.timesteps)
        if i0:
            timestep_seq = timestep_seq[:i0]
        return timestep_seq

    def loss(self, diffuse_func, x_0, noise=None, offset_noise_strength=None, **kwargs):
        b, c, h, w = x_0.shape
        t = torch.randint(0, self.sampler.timesteps, (b,), device=x_0.device).long()
        if noise is None:
            noise = torch.randn_like(x_0)

        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise
        if offset_noise_strength is None:
            offset_noise_strength = self.offset_noise_strength

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_0.shape[:2], device=noise.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample
        x_t = self.q_sample(x_0, t=t, noise=noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond, _ = self.model_predictions(diffuse_func, x_t, t)
                x_self_cond.detach_()

        pred = self.model_predictions(diffuse_func, x_t, t, x_self_cond)

        if self.objective == Config.PRED_Z:
            real = noise
        elif self.objective == Config.PRED_X0:
            real = x_0
        elif self.objective == Config.PRED_V:
            real = self.predict_v(x_0, t, noise)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = F.mse_loss(pred, real, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        return self.predict_x_t(x0, t, noise)

    def forward(self, diffuse_func, x_t, i0=None, callback_fn=None, **kwargs):
        timestep_seq = self.make_timesteps(i0)
        x_0 = None
        if callback_fn:
            callback_fn(x_t, self.timesteps)

        # t: T-1 -> 0
        for t in reversed(timestep_seq):
            self_cond = x_0 if self.self_condition else None
            x_t, x_0 = self.p_sample(diffuse_func, x_t, t, self_cond, **kwargs)
            if callback_fn:
                callback_fn(x_t, t)

        return x_t

    def scale_xt(self, x_t):
        return x_t

    def p_sample(self, diffuse_func, x_t, t: int, x_self_cond=None, clip_denoised=True, **kwargs):
        noise = torch.randn_like(x_t) if t > 0 else 0.  # no noise if t == 0

        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        x_0, _ = self.model_predictions(diffuse_func, x_t, t, x_self_cond)

        if clip_denoised:
            x_0.clamp_(-1., 1.)

        # u(x_t, x_0) = (b * \sqrt ca_{t-1} / (1-ca_t)) * x0 + ((1-ca_{t-1}) * \sqrt a_t / (1 - ca_t)) * x_t
        model_mean = extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t

        # # s(x_t, t)^2 = b * (1-ca_{t-1}) / (1-ca_t)
        # posterior_variance = extract(self.posterior_variance, t, x_t.shape)

        # \log s(x_t, t)^2 = 2 \log s(x_t, t)
        model_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        # x_{t-1} = u(x_t, x_0) + exp(0.5 * 2 * \log s(x_t, t)) * z_t
        # where, exp((0.5 * 2 * \log s(x_t, t))) = s(x_t, t), z~N(0, 1)
        x_t_1 = model_mean + (0.5 * model_log_variance).exp() * noise
        return x_t_1, x_0

    def model_predictions(
            self, diffuse_func, x_t, t, x_self_cond=None,
            clip_x_start=False, return_pred_noise=False, rederive_pred_noise=False, **kwargs):
        """x_t, t -> pred_z_t, x_0"""
        model_output = diffuse_func(x_t, t, x_self_cond=x_self_cond, **kwargs)

        pred_noise = None
        if self.objective == Config.PRED_Z:
            pred_noise = model_output
            x_0 = self.predict_start_from_noise(x_t, t, pred_noise)
            if clip_x_start:
                x_0 = torch.clamp(x_0, min=-1., max=1.)

            if return_pred_noise and clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x_t, t, x_0)

        elif self.objective == Config.PRED_X0:
            x_0 = model_output
            if clip_x_start:
                x_0 = torch.clamp(x_0, min=-1., max=1.)
            if return_pred_noise:
                pred_noise = self.predict_noise_from_start(x_t, t, x_0)

        elif self.objective == Config.PRED_V:
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

    def predict_x_t(self, x_0, t, noise):
        # x_t = x_0 * \sqrt ca_t + z_t * \sqrt (1 - ca_t)
        return extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise


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


class UNetModel(nn.Module):
    """base on Unet, add attention, res, etc."""

    def __init__(self, unit_dim,
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
            sin_pos_emb = SinusoidalEmbedding(unit_dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = unit_dim

        time_emb_dim = unit_dim * 4
        self.time_embed = nn.Sequential(
            sin_pos_emb,
            Linear(fourier_dim, time_emb_dim, mode='la', act=nn.GELU()),
            Linear(time_emb_dim, time_emb_dim, mode='l'),
        )

        num_stages = len(dim_factors)

        if isinstance(attn_heads, int):
            attn_heads = [attn_heads] * num_stages

        if isinstance(attn_dim_heads, int):
            attn_dim_heads = [attn_dim_heads] * num_stages

        make_res = partial(ResnetBlock, groups=resnet_block_groups, time_emb_dim=time_emb_dim)

        out_ch = unit_dim
        in_ch = out_ch
        self.downs = nn.ModuleList([])
        for i in range(num_stages):
            is_bottom = i == num_stages - 1
            out_ch = unit_dim * dim_factors[i]

            self.downs.append(nn.ModuleList([
                make_res(in_ch, in_ch),
                make_res(in_ch, in_ch),
                RMSNorm(in_ch),
                self.make_attn(in_ch, head_dim=attn_dim_heads[i], n_heads=attn_heads[i], is_bottom=is_bottom),
                nn.Conv2d(in_ch, out_ch, 3, padding=1) if is_bottom else Downsample(in_ch, out_ch),
            ]))

            in_ch = out_ch

        self.mid_block1 = make_res(out_ch, out_ch)
        self.mid_attn = attentions.CrossAttention3D(out_ch, n_heads=attn_heads[-1], head_dim=attn_dim_heads[-1])
        self.mid_block2 = make_res(out_ch, out_ch)

        self.ups = nn.ModuleList([])
        for i in reversed(range(num_stages)):
            is_top = i == 0
            is_bottom = i == num_stages - 1
            in_ch = unit_dim if i == 0 else unit_dim * dim_factors[i - 1]

            self.ups.append(nn.ModuleList([
                make_res(out_ch + in_ch, out_ch),
                make_res(out_ch + in_ch, out_ch),
                RMSNorm(out_ch),
                self.make_attn(out_ch, head_dim=attn_dim_heads[i], n_heads=attn_heads[i], is_bottom=is_bottom),
                nn.Conv2d(out_ch, in_ch, 3, padding=1) if is_top else Upsample(out_ch, in_ch),
            ]))

            out_ch = in_ch

        self.final_res_block = make_res(in_ch * 2, in_ch)

    def make_attn(self, in_ch, n_heads, head_dim, is_bottom, n_mem_size=4):
        attn = attentions.CrossAttention3D if is_bottom else partial(
            attentions.LinearAttention3D,
            attend=attentions.LearnedMemoryLinearAttend(n_heads, head_dim, n_mem_size),
            out_fn=Conv(n_heads * head_dim, n_heads * head_dim, 1, mode='cn', norm=RMSNorm(in_ch)),
        )
        return attn(in_ch, head_dim=head_dim, n_heads=n_heads)

    def forward(self, x, time):
        r = x.clone()
        t = self.time_embed(time)
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


class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, groups=8, drop_prob=0.,
                 use_scale_shift_norm=False, use_checkpoint=False):
        super().__init__()

        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_checkpoint = use_checkpoint

        self.in_layers = Conv(in_ch, out_ch, 3, p=1, mode='nac', norm=make_norm_fn(groups, in_ch), act=nn.SiLU())
        self.norm = make_norm_fn(groups, out_ch)
        self.emb_layers = Linear(time_emb_dim, out_ch * 2 if use_scale_shift_norm else out_ch, mode='al', act=nn.SiLU()) if time_emb_dim else None

        self.out_layers = Conv(out_ch, out_ch, 3, p=1, mode='adc', act=nn.SiLU(), drop_prob=drop_prob)
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        if use_checkpoint:
            self.forward = partial(torch_utils.ModuleManager.checkpoint, self, self.forward, is_first_layer=True)

    def forward(self, x, time_emb=None):
        h = self.in_layers(x)

        if self.emb_layers is not None and time_emb is not None:
            time_emb = self.emb_layers(time_emb)
            time_emb = time_emb[:, :, None, None]
            if self.use_scale_shift_norm:
                scale, shift = time_emb.chunk(2, dim=1)
                h = self.norm(h)
                h = h * (scale + 1) + shift
            else:
                h = h + time_emb
                h = self.norm(h)

        h = self.out_layers(h)
        return h + self.proj(x)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)
