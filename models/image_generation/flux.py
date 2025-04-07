import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn

from utils import torch_utils
from . import VAE
from .k_diffusion import EpsScaling, EulerSampler, Schedule, extract
from .. import attentions, bundles, normalizations
from ..embeddings import SinusoidalEmbedding
from ..layers import Linear
from ..multimodal_pretrain import CLIP
from ..normalizations import RMSNorm2D
from ..text_pretrain import T5
from ..text_pretrain.transformers import PositionWiseFeedForward


class Config(bundles.Config):
    backbone = dict(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth_double_blocks=19,
        depth_single_blocks=38,
        axes_dim=[16, 56, 56],
        guidance_embed=True,
    )

    t5_xxl = dict(
        hidden_size=4096,
        ff_hidden_size=10240,
        ff_act_type='FastGELU',
        num_hidden_layers=24,
        num_attention_heads=64,
        is_gated_act=True,
    )

    clip = dict(
        is_proj=False,
        **CLIP.Config.openai_text_large
    )

    vae = dict(
        use_quant_conv=False,
        use_post_quant_conv=False,
        scale_factor=0.3611,
        shift_factor=0.1159,
        backbone_config=dict(
            z_ch=16,
            ch_mult=(1, 2, 4, 4),
            attn_layers=[]
        )
    )

    default_model = 'dev'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            'dev': dict(
                t5_config=cls.t5_xxl,
                clip_config=cls.clip,
                backbone_config=cls.backbone,
                vae_config=cls.vae,
            )
        }


class WeightConverter:
    backbone_convert_dict = {
        'double_blocks.{0}_attn.norm.{1}_norm.scale': 'double_blocks.{0}_stream.{1}_norm.weight',
        'double_blocks.{0}_attn.{1}.': 'double_blocks.{0}_stream.{1}.',
        'double_blocks.{0}_mlp.0': 'double_blocks.{0}_stream.mlp.0.linear',
        'double_blocks.{0}_mlp.2': 'double_blocks.{0}_stream.mlp.1.linear',
        'double_blocks.{0}_mod.lin': 'double_blocks.{0}_stream.mod.lin',

        'single_blocks.{0}.linear1': 'single_blocks.{0}.stream.qkv',
        'single_blocks.{0}.linear2': 'single_blocks.{0}.stream.proj',
        'single_blocks.{0}.modulation.lin': 'single_blocks.{0}.stream.mod.lin',
        'single_blocks.{0}.norm.key_norm.scale': 'single_blocks.{0}.stream.key_norm.weight',
        'single_blocks.{0}.norm.query_norm.scale': 'single_blocks.{0}.stream.query_norm.weight',

        '{0}.in_layer': '{0}.0.linear',
        '{0}.out_layer': '{0}.1.linear',
        'final_layer.adaLN_modulation.1': 'head.adaLN_modulation.linear',
        'final_layer': 'head',
    }

    @classmethod
    def from_official(cls, state_dicts):
        """

        Args:
            state_dicts:
                {
                    "t5": tensors,
                    "clip": tensors,
                    "flux": tensors,
                    "vae": tensors,
                }

        """
        state_dict = OrderedDict()

        _state_dict = T5.WeightConverter.from_hf(state_dicts['t5'])
        state_dict.update({'t5.' + k: v for k, v in _state_dict.items()})

        _state_dict = CLIP.WeightConverter.from_openai(state_dicts['clip'])
        state_dict.update({'clip.' + k: v for k, v in _state_dict.items()})

        _state_dict = torch_utils.Converter.convert_keys(state_dicts['flux'], cls.backbone_convert_dict)
        state_dict.update({'backbone.' + k: v for k, v in _state_dict.items()})

        _state_dict = VAE.WeightConverter.from_ldm_official(state_dicts['vae'])
        state_dict.update({'vae.' + k: v for k, v in _state_dict.items()})

        return state_dict


class Model(nn.Module):
    """https://github.com/black-forest-labs/flux"""
    timesteps = 20
    guidance = 3.5

    image_size = (768, 768)

    use_half = True
    low_memory_run = True

    def __init__(self, t5_config=Config.t5_xxl, clip_config=Config.clip, backbone_config=Config.backbone, vae_config=Config.vae, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        self.t5 = T5.Model(**t5_config)
        self.t5.set_encoder_only()  # only for inference

        self.clip = CLIP.TextModel(**clip_config)
        self.clip.encode = self.clip.__call__

        self.backbone = Flux(**backbone_config)

        self.vae = VAE.Model(**vae_config)
        self.vae.set_inference_only()

        self.sampler = FluxSampler(schedule=FluxSchedule, scaling=FluxScaling, schedule_config=dict(num_steps=20))

    _device = None
    _dtype = None

    @property
    def device(self):
        return torch_utils.ModuleInfo.possible_device(self) if self._device is None else self._device

    @property
    def dtype(self):
        return torch_utils.ModuleInfo.possible_dtype(self) if self._dtype is None else self._dtype

    @property
    def flow_in_ch(self):
        return self.vae.z_ch

    @property
    def flow_in_size(self):
        return (
            2 * math.ceil(self.image_size[0] / 16),
            2 * math.ceil(self.image_size[1] / 16),
        )

    def set_low_memory_run(self):
        # Not critical to run single batch for decoding strategy, but reduce more GPU memory
        self.vae.encode = partial(torch_utils.ModuleManager.single_batch_run, self.vae, self.vae.encode)
        self.vae.decode = partial(torch_utils.ModuleManager.single_batch_run, self.vae, self.vae.decode)

        def wrap1(module, func):
            # note, device would be changed after model initialization.
            def wrap2(*args, **kwargs):
                return torch_utils.ModuleManager.low_memory_run(module, func, self.device, *args, **kwargs)

            return wrap2

        self.t5.encode = wrap1(self.t5, self.t5.encode)
        self.clip.encode = wrap1(self.clip, self.clip.encode)
        self.vae.encode = wrap1(self.vae, self.vae.encode)
        self.vae.decode = wrap1(self.vae, self.vae.decode)
        self.sampler.forward = wrap1(self.backbone, self.sampler.forward)
        self.sampler.to(self.device)

    def set_half(self):
        # note, vanilla sdxl vae can not convert to fp16, but bf16
        # or use a special vae checkpoint, e.g. https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
        dtype = torch.bfloat16

        torch_utils.ModuleManager.apply(
            self,
            lambda module: module.to(dtype),
            include=['t5', 'clip', 'backbone', 'vae'],
            exclude=[normalizations.GroupNorm32]
        )

        self.t5.encode = partial(torch_utils.ModuleManager.assign_dtype_run, self.t5, self.t5.encode, dtype, force_effect_module=False)
        self.clip.encode = partial(torch_utils.ModuleManager.assign_dtype_run, self.clip, self.clip.encode, dtype, force_effect_module=True)
        self.vae.encode = partial(torch_utils.ModuleManager.assign_dtype_run, self.vae, self.vae.encode, dtype, force_effect_module=False)
        self.vae.decode = partial(torch_utils.ModuleManager.assign_dtype_run, self.vae, self.vae.decode, dtype, force_effect_module=False)
        self.sampler.forward = partial(torch_utils.ModuleManager.assign_dtype_run, self.backbone, self.sampler.forward, dtype, force_effect_module=False)
        self.sampler.to(self.dtype)

    def forward(self, **kwargs):
        if self.training:
            raise NotImplementedError('Do not support train mode yet!')
        else:
            return self.post_process(**kwargs)

    def post_process(self, x=None, t5_text_ids=None, clip_text_ids=None, mask_x=None, **kwargs):
        if x is None or not len(x):  # txt2img
            x = self.gen_x_t(t5_text_ids.shape[0])
            z0 = None

        else:  # img2img
            x, z0, i0 = self.make_image_cond(x, noise=self.gen_x_t(t5_text_ids.shape[0]), **kwargs)
            kwargs.update(i0=i0)

        bs, c, H, W = x.shape

        txt = self.t5.encode(t5_text_ids)
        txt_ids = torch.zeros(bs, txt.shape[1], 3, device=x.device)
        vec = self.clip.encode(clip_text_ids)['pooler_output']

        z = self.sampler(self.flow, x, txt=txt, txt_ids=txt_ids, vec=vec, **kwargs)

        if x is not None and len(x) and mask_x is not None and len(mask_x):
            # todo: apply for different conditioning_key
            mask_x = F.interpolate(mask_x, size=z.shape[-2:])
            z = z0 * mask_x + z * (1 - mask_x)

        images = self.vae.decode(z)

        return images

    def flow(self, img, t_vec, txt, txt_ids, vec, img_cond=None, **kwargs):
        bs, c, H, W = img.shape
        h = H // 2
        w = W // 2

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

        img_ids = torch.zeros(h, w, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
        img_ids = img_ids.to(img.device)

        guidance_vec = torch.full((img.shape[0],), self.guidance, device=img.device, dtype=img.dtype)
        t_vec = torch.full((img.shape[0],), t_vec[0], dtype=img.dtype, device=img.device)

        img = self.backbone(
            img=torch.cat((img, img_cond), dim=-1) if img_cond is not None else img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )
        img = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=2, pw=2)
        return img

    def gen_x_t(self, batch_size):
        return torch.randn(
            (batch_size, self.flow_in_ch, *self.flow_in_size[::-1]),
            device=self.device, dtype=self.dtype,
            generator=torch.Generator(device=self.device),  # note, necessary
        )

    def make_image_cond(self, images, i0=None, noise=None, strength=0.75, **kwargs):
        z, _, _ = self.vae.encode(images)
        x0 = z

        if i0 is None:
            i0 = int(strength * self.sampler.num_steps)

        timestep_seq = self.sampler.make_timesteps(i0)
        t = timestep_seq[-1]
        t = torch.full((x0.shape[0],), t, device=x0.device, dtype=torch.long)
        xt = self.sampler.q_sample(x0, t, noise=noise)
        return xt, z, i0


class FluxSampler(EulerSampler):
    def p_sample(self, diffuse_func, x_t, t, prev_t=None, **diffuse_kwargs):
        # todo: add more sample methods
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_t, device=x_t.device, dtype=torch.long)

        sigma = extract(self.schedule.sigmas, t, x_t.shape)
        next_sigma = extract(self.schedule.sigmas, prev_t, x_t.shape)

        gamma = torch.where(
            torch.logical_and(self.s_tmin <= sigma, sigma <= self.s_tmax),
            min(self.s_churn / (self.schedule.num_steps - 1), 2 ** 0.5 - 1),
            0.
        ).to(sigma)

        sigma_hat = sigma * (gamma + 1.0)

        if torch.any(gamma > 0):
            eps = torch.randn_like(x_t) * self.s_noise
            x_t = x_t + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5

        possible_sigma = self.schedule.sigmas[self.sigma_to_idx(sigma_hat)]
        c_skip, c_out, c_in, c_noise = self.scaling(possible_sigma)
        c_skip, c_out, c_in = c_skip[:, None, None, None], c_out[:, None, None, None], c_in[:, None, None, None]

        d = diffuse_func(c_in * x_t, c_noise, **diffuse_kwargs) * c_out + x_t * c_skip

        d = (x_t - d) / sigma_hat
        dt = next_sigma - sigma_hat
        x_t = x_t + d * dt
        return x_t, None

    def predict_x_t(self, x_0, t, noise):
        sigma = extract(self.schedule.sigmas, t, x_0.shape)
        return x_0 * (1 - sigma) + noise * sigma


class FluxScaling(EpsScaling):
    def make_c_in(self, sigma):
        return torch.ones_like(sigma, device=sigma.device)


class FluxSchedule(Schedule):
    def make_sigmas(self):
        mu = 1.15
        sigma = 1.
        timesteps = (torch.arange(1, self.timesteps + 1, 1) / self.timesteps)
        sigmas = math.exp(mu) / (math.exp(mu) + (1 / timesteps - 1) ** sigma)
        sigmas = torch.cat([sigmas.new_zeros([1]), sigmas])
        return sigmas


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(
            self,
            in_channels, out_channels,
            hidden_size, num_heads,
            axes_dim, vec_in_dim, guidance_embed,
            context_in_dim, mlp_ratio,
            depth_double_blocks, depth_single_blocks
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.guidance_embed = guidance_embed

        self.time_embed = SinusoidalEmbedding(256, factor=1000.0)

        self.img_in = nn.Linear(self.in_channels, self.hidden_size)
        self.txt_in = nn.Linear(context_in_dim, self.hidden_size)

        self.time_in = MLPEmbedder(256, hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, hidden_size)
        self.guidance_in = MLPEmbedder(256, hidden_size) if guidance_embed else nn.Identity()

        self.embedding = FluxRotaryEmbedding(axes_dim)
        attend = FluxRotaryAttendWrapper(
            embedding=self.embedding,
            base_layer=attentions.FlashAttend()
        )

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    attend=attend
                )
                for _ in range(depth_double_blocks)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=mlp_ratio,
                    attend=attend
                )
                for _ in range(depth_single_blocks)
            ]
        )

        self.head = Head(self.hidden_size, 1, self.out_channels)

    def forward(self, img, timesteps, img_ids, txt, txt_ids, y, guidance) -> Tensor:
        # running on sequences img
        img = self.img_in(img)
        txt = self.txt_in(txt)
        vec = self.time_in(self.time_embed(timesteps))
        if self.guidance_embed:
            assert guidance is not None, ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(self.time_embed(guidance))
        vec = vec + self.vector_in(y)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.embedding.make_weights(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1]:, ...]

        img = self.head(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


class MLPEmbedder(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__(
            Linear(in_dim, hidden_dim, mode='la', act=nn.SiLU()),
            Linear(hidden_dim, hidden_dim, mode='l'),
        )


class FluxRotaryEmbedding(nn.Module):
    def __init__(self, embedding_dims, theta=10000.):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.theta = theta

    @property
    def div_terms(self):
        # note, support for meta device init
        div_terms = []
        for embedding_dim in self.embedding_dims:
            div_term = (torch.arange(0, embedding_dim, 2).float() * -(math.log(self.theta) / embedding_dim)).exp()
            div_terms.append(div_term)

        return div_terms

    def make_weights(self, positions):
        div_terms = self.div_terms
        a = []
        for i in range(len(div_terms)):
            div_term = div_terms[i]
            position = positions[..., i].float()
            div_term = div_term.to(position)
            out = torch.einsum("...n,d->...nd", position, div_term)
            out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
            out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
            a.append(out)
        weights = torch.cat(a, dim=-3)
        weights = weights.unsqueeze(1)
        return weights

    def forward(self, x, positions=None, weights=None):
        if weights is None:
            weights = self.make_weights(positions)

        x_ = x.float().reshape(*x.shape[:-1], -1, 1, 2)
        y = weights[..., 0] * x_[..., 0] + weights[..., 1] * x_[..., 1]

        return y.reshape(*x.shape).type_as(x)


class FluxRotaryAttendWrapper(nn.Module):
    def __init__(self, embedding=None, base_layer=None, base_layer_fn=None, **base_layer_kwargs):
        super().__init__()
        self.embedding = embedding
        self.base_layer = base_layer or base_layer_fn(**base_layer_kwargs)

    def forward(self, q, k, v, embedding_kwargs=dict(), **attend_kwargs):
        """
        in(q|k|v): (b n s*d)
        out(attn): (b n s*d)
        """
        q = self.embedding(q, **embedding_kwargs)
        k = self.embedding(k, **embedding_kwargs)
        attn = self.base_layer(q, k, v, **attend_kwargs)
        attn = rearrange(attn, "B H L D -> B L (H D)")
        return attn


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, attend=None):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_stream = CondStreamBlock(hidden_size, num_heads, mlp_ratio, double=True)
        self.txt_stream = CondStreamBlock(hidden_size, num_heads, mlp_ratio, double=True)
        self.attend = attend

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor) -> tuple[Tensor, Tensor]:
        txt_mod1, txt_mod2, txt_q, txt_k, txt_v, _ = self.txt_stream.stream_in(txt, vec)
        img_mod1, img_mod2, img_q, img_k, img_v, _ = self.img_stream.stream_in(img, vec)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = self.attend(q, k, v, embedding_kwargs=dict(weights=pe))
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        txt = self.txt_stream.stream_out(txt, txt_attn, txt_mod1, txt_mod2)
        img = self.img_stream.stream_out(img, img_attn, img_mod1, img_mod2)

        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attend=None):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.stream = CondStreamBlock(hidden_size, num_heads, mlp_ratio, double=False)
        self.attend = attend

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod1, _, q, k, v, mlp = self.stream.stream_in(x, vec)
        attn = self.attend(q, k, v, embedding_kwargs=dict(weights=pe))
        attn = torch.cat((attn, mlp), 2)
        x = self.stream.stream_out(x, attn, mod1)
        return x


class CondStreamBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, double):
        super().__init__()
        self.double = double
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        if double:
            qkv_output_size = hidden_size * 3
            proj_input_size = hidden_size
        else:
            qkv_output_size = hidden_size * 3 + mlp_hidden_size
            proj_input_size = hidden_size + mlp_hidden_size
        self.hidden_size = hidden_size
        self.mlp_hidden_size = mlp_hidden_size

        self.mod = Modulation(hidden_size, double=double)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_size, qkv_output_size)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.query_norm = RMSNorm2D(head_dim)
        self.key_norm = RMSNorm2D(head_dim)

        self.proj = nn.Linear(proj_input_size, hidden_size)

        if double:
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.mlp = PositionWiseFeedForward(hidden_size, mlp_hidden_size, act=self.mlp_act, drop_prob=0)

    def stream_in(self, x, vec):
        mod1, mod2 = self.mod(vec)
        x = self.norm1(x)
        modulated = (1 + mod1.scale) * x + mod1.shift
        if self.double:
            qkv = self.qkv(modulated)
            mlp = None
        else:
            qkv, mlp = torch.split(self.qkv(modulated), [3 * self.hidden_size, self.mlp_hidden_size], dim=-1)
            mlp = self.mlp_act(mlp)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q = self.query_norm(q)
        k = self.key_norm(k)
        return mod1, mod2, q, k, v, mlp

    def stream_out(self, x, attn, mod1, mod2=None):
        x = x + mod1.gate * self.proj(attn)
        if self.double:
            x = x + mod2.gate * self.mlp((1 + mod2.scale) * self.norm2(x) + mod2.shift)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class Head(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.adaLN_modulation = Linear(hidden_size, 2 * hidden_size, mode='al', act=nn.SiLU())
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
