import math
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from utils import torch_utils
from .. import activations, attentions, bundles, embeddings, normalizations
from ..embeddings import SinusoidalEmbedding
from ..image_generation import VAE, k_diffusion, sdv1
from ..layers import Linear
from ..multimodal_pretrain import CLIP
from ..text_pretrain import T5, transformers


class Config(bundles.Config):
    class ModelType:
        t2v = 't2v'
        i2v = 'i2v'
        flf2v = 'flf2v'

    backbone_14b = dict(
        hidden_size=5120,
        ff_hidden_size=13824,
        num_heads=40,
        num_blocks=40,
    )

    backbone_t2v_1_3b = dict(
        in_ch=16,
        hidden_size=1536,
        ff_hidden_size=8960,
        num_heads=12,
        num_blocks=30,
        model_type=ModelType.t2v
    )

    backbone_t2v_14b = dict(
        in_ch=16,
        model_type=ModelType.t2v,
        **backbone_14b
    )

    backbone_i2v_14b = dict(
        in_ch=36,
        model_type=ModelType.i2v,
        **backbone_14b
    )

    backbone_flf2v_14b = dict(
        in_ch=36,
        model_type=ModelType.flf2v,
        **backbone_14b
    )

    t5_xxl = dict(
        vocab_size=256384,
        hidden_size=4096,
        ff_hidden_size=10240,
        num_attention_heads=64,
        num_hidden_layers=24,
        share_rel_weights=False,
        is_gated_act=True,
        ff_act_type='FastGELU',
    )

    clip = dict(
        output_size=1024,
        hidden_size=1280,
        image_size=224,
        num_attention_heads=16,
        num_hidden_layers=32,
        ff_ratio=4.,
        patch_size=14,
        act_type='GELU',
        attend_type='WanFlashAttend',
        separate=False,
        patch_bias=False,
        layer_idx=32 - 2,  # the last two layers
    )

    vae = dict(
        backbone_config=dict(
            z_ch=16,
            unit_ch=96,
            ch_mult=(1, 2, 4, 4),
            attn_layers=[]
        )
    )

    sampler = dict(
        schedule='FlowUniPCMultistepSchedule',
        scaling='XScaling',
        schedule_config=dict(num_steps=50)
    )

    default_model = 't2v-1.3b'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            't2v-1.3b': dict(
                t5_config=cls.t5_xxl,
                backbone_config=cls.backbone_t2v_1_3b,
                vae_config=cls.vae,
                model_type=cls.ModelType.t2v,
            ),

            't2v-14b': dict(
                t5_config=cls.t5_xxl,
                backbone_config=cls.backbone_t2v_14b,
                vae_config=cls.vae,
                model_type=cls.ModelType.t2v,
            ),

            'i2v_14b': dict(
                t5_config=cls.t5_xxl,
                backbone_config=cls.backbone_i2v_14b,
                vae_config=cls.vae,
                clip_config=cls.clip,
                model_type=cls.ModelType.i2v,
            ),

            'flf2v_14b': dict(
                t5_config=cls.t5_xxl,
                backbone_config=cls.backbone_flf2v_14b,
                vae_config=cls.vae,
                clip_config=cls.clip,
                model_type=cls.ModelType.flf2v,
            ),
        }


class WeightConverter:
    t5_convert_dict = {
        'token_embedding': 'embedding',

        'blocks.{0}.norm1': 'encoder.{0}.attn_res.norm',
        'blocks.{0}.attn.q': 'encoder.{0}.attn_res.fn.to_qkv.0',
        'blocks.{0}.attn.k': 'encoder.{0}.attn_res.fn.to_qkv.1',
        'blocks.{0}.attn.v': 'encoder.{0}.attn_res.fn.to_qkv.2',
        'blocks.{0}.attn.o': 'encoder.{0}.attn_res.fn.to_out.linear',

        'blocks.{0}.norm2': 'encoder.{0}.ff_res.norm',
        'blocks.{0}.ffn.gate.0': 'encoder.{0}.ff_res.fn.f1.linear',
        'blocks.{0}.ffn.fc1': 'encoder.{0}.ff_res.fn.f3.linear',
        'blocks.{0}.ffn.fc2': 'encoder.{0}.ff_res.fn.f2.linear',

        'blocks.{0}.pos_embedding.embedding': 'encoder.{0}.attn_res.fn.attend.relative_bias',

        'norm': 'encoder_norm'
    }

    vae_convert_dict = {
        '{0}.conv1': '{0}.conv_in',
        '{0}.{1}samples.{2}.residual.{3}.': '{0}.{1}.{2}.fn.{3}.',
        '{0}.{1}samples.{2}.shortcut': '{0}.{1}.{2}.proj',
        '{0}.{1}samples.{2}.resample.1': '{0}.{1}.{2}.resample.1',
        '{0}.{1}samples.{2}.time_conv': '{0}.{1}.{2}.time_conv',

        '{0}.middle.{1}.residual.{2}.': '{0}.neck.{1}.fn.{2}.',
        '{0}.middle.{1}.norm': '{0}.neck.{1}.0',
        '{0}.middle.{1}.to_qkv': '{0}.neck.{1}.1.to_qkv',
        '{0}.middle.{1}.proj': '{0}.neck.{1}.1.to_out',

        '{0}.head.0': '{0}.head.0',
        '{0}.head.2': '{0}.head.2',

        'conv1': 'quant_conv',
        'conv2': 'post_quant_conv',
    }

    clip_convert_dict = {
        'textual.token_embedding': 'text_model.embedding.token',
        'textual.type_embedding': 'text_model.embedding.type_embedding',
        'textual.pos_embedding': 'text_model.embedding.position',
        'textual.norm': 'text_model.norm',
        'textual.blocks.{0}.attn.q': 'text_model.encoder.{0}.attn_res.fn.to_qkv.0',
        'textual.blocks.{0}.attn.k': 'text_model.encoder.{0}.attn_res.fn.to_qkv.1',
        'textual.blocks.{0}.attn.v': 'text_model.encoder.{0}.attn_res.fn.to_qkv.2',
        'textual.blocks.{0}.attn.o': 'text_model.encoder.{0}.attn_res.fn.to_out.linear',
        'textual.blocks.{0}.norm1': 'text_model.encoder.{0}.attn_res.norm',
        'textual.blocks.{0}.ffn.0': 'text_model.encoder.{0}.ff_res.fn.0.linear',
        'textual.blocks.{0}.ffn.2': 'text_model.encoder.{0}.ff_res.fn.1.linear',
        'textual.blocks.{0}.norm2': 'text_model.encoder.{0}.ff_res.norm',
        'textual.head.0': 'text_model.proj.0',
        'textual.head.2': 'text_model.proj.2',

        'visual.cls_embedding': 'vision_model.embedding.cls',
        'visual.pos_embedding': 'vision_model.embedding.position.weight',
        'visual.patch_embedding': 'vision_model.embedding.patch.fn.0',
        'visual.pre_norm': 'vision_model.norm1',
        'visual.post_norm': 'vision_model.norm2',
        'visual.transformer.{0}.attn.to_qkv': 'vision_model.encoder.{0}.attn_res.fn.to_qkv',
        'visual.transformer.{0}.attn.proj': 'vision_model.encoder.{0}.attn_res.fn.to_out.linear',
        'visual.transformer.{0}.norm1': 'vision_model.encoder.{0}.attn_res.norm',
        'visual.transformer.{0}.norm2': 'vision_model.encoder.{0}.ff_res.norm',
        'visual.transformer.{0}.mlp.0': 'vision_model.encoder.{0}.ff_res.fn.0.linear',
        'visual.transformer.{0}.mlp.2': 'vision_model.encoder.{0}.ff_res.fn.1.linear',
    }

    backbone_convert_dict = {
        'time_embedding.0': 'time_embed.1.linear',
        'time_embedding.2': 'time_embed.2.linear',

        'blocks.{0}.self_attn.q': 'blocks.{0}.attn.to_qkv.0.linear',
        'blocks.{0}.self_attn.norm_q': 'blocks.{0}.attn.to_qkv.0.norm',
        'blocks.{0}.self_attn.k': 'blocks.{0}.attn.to_qkv.1.linear',
        'blocks.{0}.self_attn.norm_k': 'blocks.{0}.attn.to_qkv.1.norm',
        'blocks.{0}.self_attn.v': 'blocks.{0}.attn.to_qkv.2.linear',
        'blocks.{0}.self_attn.o': 'blocks.{0}.attn.to_out.linear',

        'blocks.{0}.cross_attn.q': 'blocks.{0}.de_attn.to_qkv.0.linear',
        'blocks.{0}.cross_attn.norm_q': 'blocks.{0}.de_attn.to_qkv.0.norm',
        'blocks.{0}.cross_attn.k': 'blocks.{0}.de_attn.to_qkv.1.linear',
        'blocks.{0}.cross_attn.norm_k': 'blocks.{0}.de_attn.to_qkv.1.norm',
        'blocks.{0}.cross_attn.v': 'blocks.{0}.de_attn.to_qkv.2.linear',
        'blocks.{0}.cross_attn.o': 'blocks.{0}.de_attn.to_out.linear',

        'blocks.{0}.ffn.0': 'blocks.{0}.ff_proj.0',
        'blocks.{0}.ffn.2': 'blocks.{0}.ff_proj.2',
        'blocks.{0}.norm3': 'blocks.{0}.de_attn_norm',  # note, norm3 is de_attn_norm not norm2!!!
    }

    @classmethod
    def from_official(cls, state_dicts):
        state_dict = OrderedDict()

        if 't5' in state_dicts:
            _state_dict = torch_utils.Converter.convert_keys(state_dicts['t5'], cls.t5_convert_dict)
            state_dict.update({'t5.' + k: v for k, v in _state_dict.items()})

        if 'clip' in state_dicts:
            _state_dict = torch_utils.Converter.convert_keys(state_dicts['clip'], cls.clip_convert_dict)
            if 'visual.head' in _state_dict:
                _state_dict['vision_model.proj.weight'] = _state_dict.pop('visual.head').T
            elif 'visual.head.weight' in _state_dict:
                _state_dict['vision_model.proj.weight'] = _state_dict.pop('visual.head.weight')
            _state_dict['vision_model.embedding.cls'] = _state_dict['vision_model.embedding.cls'][0, 0]
            _state_dict['vision_model.embedding.position.weight'] = _state_dict['vision_model.embedding.position.weight'][0]
            state_dict.update({'clip.' + k: v for k, v in _state_dict.items()})

        if 'vae' in state_dicts:
            _state_dict = torch_utils.Converter.convert_keys(state_dicts['vae'], cls.vae_convert_dict)
            state_dict.update({'vae.' + k.replace('gamma', 'weight'): v for k, v in _state_dict.items()})

        if 'backbone' in state_dicts:
            _state_dict = torch_utils.Converter.convert_keys(state_dicts['backbone'], cls.backbone_convert_dict)
            state_dict.update({'backbone.' + k: v for k, v in _state_dict.items()})

        return state_dict


class Model(nn.Module):
    """https://github.com/Wan-Video/Wan2.1"""
    timesteps = 20
    guidance = 3.5

    image_size = (832, 480)
    num_frame = 81  # fps=16, video_dur=5s -> frame_size = 5 * 16 = 81

    model_type = Config.ModelType.t2v

    def __init__(self, t5_config=Config.t5_xxl, clip_config=Config.clip, backbone_config=Config.backbone_t2v_1_3b, vae_config=Config.vae, sampler_config=Config.sampler, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        self.t5 = T5.Model(**t5_config)
        self.t5.set_encoder_only()  # only for inference

        if self.model_type != Config.ModelType.t2v:
            self.clip = CLIPEmbedder(**clip_config)
        else:
            self.clip = nn.Identity()
        self.clip.encode = self.clip.__call__

        self.backbone = WanUNetModel(**backbone_config)

        self.vae = WanVAE(**vae_config)
        self.vae.set_inference_only()

        self.sampler = k_diffusion.EulerSampler(**sampler_config)

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
        dtype = torch.bfloat16

        torch_utils.ModuleManager.apply(
            self,
            lambda module: module.to(dtype),
            include=['t5', 'clip', 'backbone', 'vae'],
            exclude=[
                normalizations.LayerNorm32,
                normalizations.GroupNorm32,
                normalizations.RMSNorm,
                'backbone.freqs', 'backbone.time_embed', 'backbone.time_projection', WanRotaryEmbedding,
            ]
        )

        self.t5.encode = partial(torch_utils.ModuleManager.assign_dtype_run, self.t5, self.t5.encode, dtype, force_effect_module=False)
        self.clip.encode = partial(torch_utils.ModuleManager.assign_dtype_run, self.clip, self.clip.encode, dtype, force_effect_module=True)
        self.vae.encode = partial(torch_utils.ModuleManager.assign_dtype_run, self.vae, self.vae.encode, dtype, force_effect_module=False)
        self.vae.decode = partial(torch_utils.ModuleManager.assign_dtype_run, self.vae, self.vae.decode, dtype, force_effect_module=False)
        self.sampler.forward = partial(torch_utils.ModuleManager.assign_dtype_run, self.backbone, self.sampler.forward, dtype, force_effect_module=False)
        self.sampler.to(self.dtype)

    _device = None
    _dtype = None

    @property
    def device(self):
        return torch_utils.ModuleInfo.possible_device(self) if self._device is None else self._device

    @property
    def dtype(self):
        return torch_utils.ModuleInfo.possible_dtype(self) if self._dtype is None else self._dtype

    def forward(self, **kwargs):
        if self.training:
            raise NotImplementedError('Do not support train mode yet!')
        else:
            return self.inference(**kwargs)

    def inference(self, x=None, text_ids=None, image_size=None, num_frame=81, **kwargs):
        b = len(text_ids)

        txt_cond = self.make_txt_cond(text_ids, **kwargs)
        kwargs.update(txt_cond)

        y = None
        if self.model_type != Config.ModelType.t2v:
            y = self.make_image_cond(x, **kwargs)
            kwargs.update(y=y)

        # make x_t
        x = self.gen_x_t(b, num_frame, image_size, y=y)

        if self.model_type != Config.ModelType.t2v:
            clip_feat = self.clip.encode(x)
            kwargs.update(clip_feat=clip_feat)

        z = self.sampler(self.process, x, **kwargs)
        video = self.vae.decode(z)
        return video

    def make_txt_cond(self, text_ids, text_mask, neg_text_ids=None, neg_text_mask=None, scale=5., text_len=512, **kwargs) -> dict:
        c = self.t5.encode(text_ids, text_mask[None, None])
        seq_lens = text_mask.gt(0).sum(dim=1).long()
        c = [u[:v] for u, v in zip(c, seq_lens)]
        c = torch.stack(c)
        c = torch.cat([c, c.new_zeros(c.size(0), text_len - c.size(1), c.size(2))], dim=1)

        uc = None
        if scale > 1.0 and neg_text_ids is not None:
            uc = self.t5.encode(neg_text_ids, neg_text_mask[None, None])
            seq_lens = neg_text_mask.gt(0).sum(dim=1).long()
            uc = [u[:v] for u, v in zip(uc, seq_lens)]
            uc = torch.stack(uc)
            uc = torch.cat([uc, uc.new_zeros(uc.size(0), text_len - uc.size(1), uc.size(2))], dim=1)

        return dict(
            cond=c,
            un_cond=uc
        )

    def make_image_cond(self, images, num_frame=None, max_area=720 * 1280, **kwargs):
        b, c, h, w = images.shape
        aspect_ratio = h / w
        lat_h = round(np.sqrt(max_area * aspect_ratio) // 8 // self.backbone.patch_size[1] * self.backbone.patch_size[1])
        lat_w = round(np.sqrt(max_area / aspect_ratio) // 8 // self.backbone.patch_size[2] * self.backbone.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]

        x = torch.concat([
            torch.nn.functional.interpolate(images[:, None].cpu(), size=(h, w), mode='bicubic').transpose(1, 2),
            torch.zeros(b, c, num_frame - 1, h, w)
        ], dim=2).to(images)  # (b, c, frame_size, h, w)
        y, _, _ = self.vae.encode(x, sample_posterior=False)
        hw = y.shape[-2:]

        mask = torch.ones(b, num_frame, *hw, device=self.device)
        mask[:, 1:] = 0  # mask all but the first frame
        mask = torch.concat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)  # Be a multiple of 4
        mask = mask.view(b, mask.shape[1] // 4, 4, *hw)
        mask = mask.transpose(1, 2)  # (b, 4, frame_size, h, w)

        y = torch.concat([mask, y], dim=1)
        return y

    def process(self, x, time, cond=None, un_cond=None, scale=5., **backbone_kwargs):
        if un_cond is not None:
            x = torch.cat([x] * 2)
            time = torch.cat([time] * 2)
            cond = torch.cat([un_cond, cond])

        z = self.backbone(x=x, t=time, context=cond, **backbone_kwargs)
        if un_cond is not None:
            e_t_uncond, e_t = z.chunk(2)
            e_t = e_t_uncond + scale * (e_t - e_t_uncond)
        else:
            e_t = z

        return e_t

    @property
    def process_in_ch(self):
        return self.vae.z_ch

    def process_in_size(self, image_size=None):
        image_size = image_size or self.image_size
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        return (
            2 * math.ceil(image_size[0] / 16),
            2 * math.ceil(image_size[1] / 16),
        )  # (w, h)

    def process_in_frame_size(self, num_frame=None):
        num_frame = num_frame or self.num_frame
        frame_size = (num_frame - 1) // 4 + 1
        return frame_size

    def gen_x_t(self, batch_size, num_frame=None, image_size=None, y=None):
        if y is None:
            shape = (self.process_in_frame_size(num_frame), *self.process_in_size(image_size)[::-1])
        else:
            shape = y.shape[2:]
        return torch.randn(
            (batch_size, self.process_in_ch, *shape),  # (b, c, t, h, w)
            device=self.device, dtype=self.dtype,
            # generator=torch.Generator(device=self.device),  # note, not necessary
        )


class WanVAE(VAE.Model):
    scale_factor = 1 / torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ])[None, :, None, None, None]
    shift_factor = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ])[None, :, None, None, None]

    def set_encoder(self, **backbone_config):
        self.encoder = WanVAEEncoder3d(self.img_ch, **backbone_config)
        self.quant_conv = WanVAECausalConv3d(self.encoder.z_ch * 2, self.encoder.z_ch * 2, 1) if self.use_quant_conv else nn.Identity()

    def set_decoder(self, **backbone_config):
        z_ch = self.encoder.z_ch
        self.post_quant_conv = WanVAECausalConv3d(z_ch, z_ch, 1) if self.use_post_quant_conv else nn.Identity()
        self.decoder = WanVAEDecoder3d(z_ch, self.img_ch, **backbone_config)

    @staticmethod
    def count_conv3d(model):
        count = 0
        for m in model.modules():
            if isinstance(m, WanVAECausalConv3d):
                count += 1
        return count

    def encode(self, x, sample_posterior=True):
        feat_cache = [None] * self.count_conv3d(self.encoder)

        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4

        out = None
        for i in range(iter_):
            feat_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=feat_cache,
                    feat_idx=feat_idx
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=feat_cache,
                    feat_idx=feat_idx
                )
                out = torch.cat([out, out_], 2)

        out = self.quant_conv(out)
        z, mean, log_var = self.re_parametrize(out, sample_posterior=sample_posterior)

        scale_factor = self.scale_factor
        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.to(z)
        shift_factor = self.shift_factor
        if isinstance(shift_factor, torch.Tensor):
            shift_factor = shift_factor.to(z)

        z = scale_factor * (z - shift_factor)
        return z, mean, log_var

    def decode(self, z):
        feat_cache = [None] * self.count_conv3d(self.decoder)

        scale_factor = self.scale_factor
        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.to(z)
        shift_factor = self.shift_factor
        if isinstance(shift_factor, torch.Tensor):
            shift_factor = shift_factor.to(z)

        z = z / scale_factor + shift_factor

        iter_ = z.shape[2]
        x = self.post_quant_conv(z)
        out = None
        for i in range(iter_):
            feat_idx = [0]
            out_ = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=feat_cache,
                feat_idx=feat_idx
            )
            if i == 0:
                out = out_
            else:
                out = torch.cat([out, out_], 2)
        return out


class WanVAECausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class WanVAEResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, conv_shortcut=False, drop_prob=0.):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.in_channels = in_ch
        self.out_channels = out_ch

        if in_ch != out_ch:
            if conv_shortcut:
                shortcut = WanVAECausalConv3d(in_ch, out_ch, 3, stride=1, padding=1)
            else:
                shortcut = WanVAECausalConv3d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            shortcut = nn.Identity()

        self.proj = shortcut
        self.fn = WanVAEResFn(in_ch, out_ch, drop_prob=drop_prob)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h1 = self.proj(x)
        h2 = self.fn(x, feat_cache, feat_idx)
        return h1 + h2


class WanVAEResFn(nn.ModuleList):
    def __init__(self, in_ch, out_ch, drop_prob=0.):
        super().__init__([
            normalizations.RMSNorm4D(in_ch),
            nn.SiLU(),
            WanVAECausalConv3d(in_ch, out_ch, 3, padding=1),

            normalizations.RMSNorm4D(out_ch),
            nn.SiLU(),
            nn.Dropout(drop_prob),
            WanVAECausalConv3d(out_ch, out_ch, 3, padding=1)
        ])

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        for layer in self:
            if isinstance(layer, WanVAECausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -2:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class WanVAEAttentionBlock(nn.Sequential):
    def __init__(self, in_ch):
        super().__init__(
            normalizations.RMSNorm3D(in_ch),
            attentions.CrossAttention3D(
                n_heads=1, head_dim=in_ch, separate=False,
                # attend=attentions.SplitScaleAttend()
            )
        )

    def forward(self, x):
        h = super().forward(x)
        return x + h


class WanVAEDownSample(nn.Module):
    def __init__(self, in_ch, is_2d=True):
        super().__init__()
        self.is_2d = is_2d

        self.resample = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(in_ch, in_ch, 3, stride=(2, 2))
        )

        if not is_2d:
            self.time_conv = WanVAECausalConv3d(in_ch, in_ch, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if not self.is_2d:
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class UpSample32(nn.Upsample):
    def forward(self, x):
        """forced to use fp32"""
        return super().forward(x.float()).type_as(x)


class WanVAEUpSample(nn.Module):
    def __init__(self, in_ch, is_2d=True):
        super().__init__()
        self.is_2d = is_2d

        self.resample = nn.Sequential(
            UpSample32(scale_factor=(2., 2.), mode='nearest-exact'),
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1)
        )
        if not is_2d:
            self.time_conv = WanVAECausalConv3d(in_ch, in_ch * 2, (3, 1, 1), padding=(1, 0, 0))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if not self.is_2d:
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -2:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x


class WanVAEEncoder3d(nn.Module):
    def __init__(
            self, in_ch, unit_ch=128, z_ch=64,
            ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2, attn_layers=(-1, -2),
            drop_prob=0.0, double_z=True,
            **ignore_kwargs
    ):
        super().__init__()
        num_layers = len(ch_mult)

        self.conv_in = WanVAECausalConv3d(in_ch, unit_ch, 3, stride=1, padding=1)

        in_ch = unit_ch
        down = []
        for i in range(num_layers):
            is_top = i == num_layers - 1
            is_2d = i == 0  # the first layers
            out_ch = unit_ch * ch_mult[i]

            for _ in range(num_res_blocks):
                down.append(WanVAEResBlock(in_ch, out_ch, drop_prob=drop_prob))
                if i in attn_layers:
                    down.append(WanVAEAttentionBlock(out_ch))

                in_ch = out_ch

            if not is_top:
                down.append(WanVAEDownSample(in_ch, is_2d))
        self.down = nn.ModuleList(down)

        self.neck = nn.ModuleList([
            WanVAEResBlock(in_ch, in_ch, drop_prob=drop_prob),
            WanVAEAttentionBlock(in_ch),
            WanVAEResBlock(in_ch, in_ch, drop_prob=drop_prob)
        ])

        out_ch = 2 * z_ch if double_z else z_ch
        self.head = nn.ModuleList([
            normalizations.RMSNorm4D(in_ch),
            nn.SiLU(),
            WanVAECausalConv3d(in_ch, out_ch, 3, padding=1),
        ])

        self.out_channels = out_ch
        self.down_scale = 2 ** (num_layers - 1)
        self.z_ch = z_ch

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -2:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for layer in self.down:
            if not isinstance(layer, WanVAEAttentionBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            elif isinstance(layer, WanVAEAttentionBlock):
                b, c, t, h, w = x.shape
                x = rearrange(x, 'b c t h w -> (b t) c h w')
                x = layer(x)
                x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
            else:
                x = layer(x)

        for layer in self.neck:
            if isinstance(layer, WanVAEResBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            elif isinstance(layer, WanVAEAttentionBlock):
                b, c, t, h, w = x.shape
                x = rearrange(x, 'b c t h w -> (b t) c h w')
                x = layer(x)
                x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, WanVAECausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -2:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class WanVAEDecoder3d(nn.Module):
    def __init__(
            self, in_ch, out_ch, unit_ch=128,
            ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2, attn_layers=(-1, -2),
            drop_prob=0.0,
            **ignore_kwargs
    ):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        num_layers = len(ch_mult)
        attn_layers = [i % num_layers for i in attn_layers]

        # z to block_in
        in_ch = unit_ch * ch_mult[num_layers - 1]
        self.conv_in = WanVAECausalConv3d(self.in_channels, in_ch, 3, stride=1, padding=1)

        # middle
        self.neck = nn.ModuleList([
            WanVAEResBlock(in_ch, in_ch, drop_prob=drop_prob),
            WanVAEAttentionBlock(in_ch),
            WanVAEResBlock(in_ch, in_ch, drop_prob=drop_prob)
        ])

        # upsample
        up = []
        for i in reversed(range(num_layers)):
            is_bottom = i == 0
            is_top = i == num_layers - 1
            is_2d = i == 1  # the first 2 layers
            if not is_top:
                in_ch //= 2
            out_ch = unit_ch * ch_mult[i]
            for j in range(num_res_blocks + 1):
                up.append(WanVAEResBlock(in_ch, out_ch, drop_prob=drop_prob))
                if i in attn_layers:
                    up.append(WanVAEAttentionBlock(out_ch))
                in_ch = out_ch

            # upsample block
            if not is_bottom:
                up.append(WanVAEUpSample(in_ch, is_2d))
        # note, implement different from ldm
        self.up = nn.ModuleList(up)

        self.head = nn.ModuleList([
            normalizations.RMSNorm4D(in_ch),
            nn.SiLU(),
            WanVAECausalConv3d(in_ch, self.out_channels, 3, padding=1),
        ])

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -2:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for layer in self.neck:
            if isinstance(layer, WanVAEResBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            elif isinstance(layer, WanVAEAttentionBlock):
                b, c, t, h, w = x.shape
                x = rearrange(x, 'b c t h w -> (b t) c h w')
                x = layer(x)
                x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
            else:
                x = layer(x)

        for layer in self.up:
            if not isinstance(layer, WanVAEAttentionBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            elif isinstance(layer, WanVAEAttentionBlock):
                b, c, t, h, w = x.shape
                x = rearrange(x, 'b c t h w -> (b t) c h w')
                x = layer(x)
                x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
            else:
                x = layer(x)

        for layer in self.head:
            if isinstance(layer, WanVAECausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -2:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class WanCLIP(CLIP.Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.text_model = WanCLIPTextTransformer(**kwargs)


class CLIPEmbedder(nn.Module):
    def __init__(self, layer_idx=None, **kwargs):
        super().__init__()
        self.vision_model = CLIP.VisionTransformer(**kwargs)
        self.callback = sdv1.Callback(layer_idx)

    def forward(self, image, **kwargs):
        x = self.vision_model.backbone(image, callback_fn=self.callback)
        z = self.callback.cache_hidden_state
        return z


class WanCLIPTextModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.text_model = WanCLIPTextTransformer(**kwargs)

    def forward(self, text_ids, **kwargs):
        return self.text_model(text_ids, **kwargs)


class WanCLIPTextTransformer(CLIP.TextTransformer):
    def __init__(
            self, vocab_size=250002, hidden_size=1024, output_size=1024,
            max_seq_len=512 + 2, num_attention_heads=16, num_hidden_layers=24, ff_ratio=4.0,
            separate=True,
            act_type='GELU', attend_type='FlashAttend',
            use_checkpoint=True
    ):
        super(CLIP.TextTransformer, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.embedding = WanCLIPDecoderEmbedding(vocab_size, hidden_size, max_seq_len=max_seq_len)
        self.encoder = transformers.TransformerSequential(
            hidden_size, num_attention_heads, int(hidden_size * ff_ratio), norm_first=True,
            attend_fn=attentions.make_attend_fn.get(attend_type),
            fn_kwargs=dict(separate=separate),
            ff_kwargs=dict(act=activations.make_act_fn.get(act_type)()),
            num_blocks=num_hidden_layers,
            use_checkpoint=use_checkpoint
        )
        self.norm = nn.LayerNorm(hidden_size)
        mid_dim = (hidden_size + output_size) // 2
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, mid_dim, bias=False),
            nn.GELU(),
            nn.Linear(mid_dim, output_size, bias=False)
        )

    def head(self, sequence, h):
        # average pooling
        mask = sequence.ne(self.pad_id).unsqueeze(-1).to(h)
        h = (h * mask).sum(dim=1) / mask.sum(dim=1)

        # head
        pooled_output = self.proj(h)
        return pooled_output


class WanCLIPDecoderEmbedding(nn.Module):
    """TokenEmbedding + PositionalEmbedding"""

    def __init__(self, vocab_size, embedding_dim, pad_id=None, max_seq_len=512, type_size=1):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        # note, different here, add type embedding specially
        self.type_embedding = nn.Embedding(type_size, embedding_dim)
        self.position = embeddings.LearnedPositionEmbedding(max_seq_len, embedding_dim)

    def forward(self, sequence):
        """(b, s) -> (b, s, h)
        note, s is a dynamic var"""
        x = (
                self.token(sequence)
                + self.type_embedding(torch.zeros_like(sequence))
                + self.position(sequence)
        )
        return x


class WanUNetModel(nn.Module):
    def __init__(
            self,
            in_ch=16, out_ch=16,
            hidden_size=2048, ff_hidden_size=8960, freq_size=256, context_in_size=4096,
            patch_size=(1, 2, 2), window_size=(-1, -1),
            num_heads=16, num_blocks=32,
            text_len=512, max_seq_len=1024,
            eps=1e-6, model_type=Config.ModelType.t2v, **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.text_len = text_len
        self.max_seq_len = max_seq_len
        self.out_ch = out_ch
        self.patch_size = patch_size

        # embeddings
        self.patch_embedding = nn.Conv3d(in_ch, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(context_in_size, hidden_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_size, hidden_size)
        )

        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(freq_size),
            Linear(freq_size, hidden_size, mode='la', act=nn.SiLU()),
            Linear(hidden_size, hidden_size, mode='l'),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6)
        )

        d = hidden_size // num_heads
        self.rot_embeddings = nn.ModuleList([
            WanRotaryEmbedding(d - 4 * (d // 6)),
            WanRotaryEmbedding(2 * (d // 6)),
            WanRotaryEmbedding(2 * (d // 6)),
        ])

        self.blocks = nn.ModuleList([WanTransformerBlock(
            hidden_size, num_heads, ff_hidden_size, window_size=window_size
        ) for _ in range(num_blocks)])

        # head
        self.head = WanUNetHead(hidden_size, out_ch, patch_size, eps)

        if model_type != Config.ModelType.t2v:
            self.img_emb = MLPProj(1280, hidden_size, flf_pos_emb=model_type == Config.ModelType.flf2v)

    def forward(self, x, t, context, clip_feat=None, y=None, **kwargs):
        """
        Forward pass through the diffusion model

        Args:
            x (Tensor):
               shape [b, C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (Tensor):
                List of text embeddings each with shape [b, L, C]
            clip_feat (Tensor, *optional*):
                CLIP image features for image-to-video mode or first-last-frame-to-video mode
            y (Tensor, *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            Tensor:
                List of denoised video tensors with original input shapes [b, C_out, F, H / 8, W / 8]
        """
        if y is not None:
            x = torch.cat([x, y], dim=1)

        # embeddings
        x = self.patch_embedding(x)
        grid_sizes = torch.stack([torch.tensor(u.shape[1:], dtype=torch.long) for u in x])
        x = x.flatten(2).transpose(1, 2)
        seq_lens = torch.tensor([u.size(0) for u in x], dtype=torch.long)

        # time embeddings
        e = self.time_embed(t.float())
        e0 = self.time_projection(e).unflatten(1, (6, self.hidden_size))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(context)

        if clip_feat is not None:
            context_clip = self.img_emb(clip_feat)  # bs x 257 (x2) x dim
            context = torch.concat([context_clip, context], dim=1)

        rot_embedding_weights = []
        for m in self.rot_embeddings:
            rot_embedding_weights.append(m.make_weights(self.max_seq_len))
        rot_embedding_weights = torch.cat(rot_embedding_weights, dim=1)

        for block in self.blocks:
            x = block(
                x,
                e=e0,
                seq_lens=seq_lens,
                context=context,
                context_lens=context_lens,
                embedding_kwargs=dict(
                    grid_sizes=grid_sizes,
                    weights=rot_embedding_weights,
                ),
            )

        # head
        x = self.head(x, e)
        x = self.unpatchify(x, grid_sizes)
        return x

    def unpatchify(self, x, grid_sizes):
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x (Tensor):
                List of patchified features, each with shape [b, L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            Tensor:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_ch
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        out = torch.stack(out)
        return out


class WanTransformerBlock(nn.Module):
    def __init__(
            self,
            hidden_size, num_heads, ff_hidden_size,
            window_size=(-1, -1), drop_prob=0.1,
            **kwargs
    ):
        super().__init__()
        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, hidden_size) / hidden_size ** 0.5)
        # layers
        self.attn_norm = normalizations.LayerNorm32(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = WanCrossAttention2D(
            n_heads=num_heads,
            model_dim=hidden_size,
            drop_prob=drop_prob,
            attend=attentions.RotaryAttendWrapper(
                dim=hidden_size,
                base_layer_fn=WanFlashAttend,
                window_size=window_size,
                embedding=WanRotaryEmbedding(1)  # weights will be counted before, so do not need weights here
            )
        )
        self.de_attn_norm = normalizations.LayerNorm32(hidden_size, elementwise_affine=True, eps=1e-6)
        self.de_attn = WanCrossAttention2D(
            n_heads=num_heads,
            model_dim=hidden_size,
            drop_prob=drop_prob,
            attend=WanFlashAttend()
        )
        self.ff_proj = nn.Sequential(
            nn.Linear(hidden_size, ff_hidden_size),
            nn.GELU(approximate='tanh'),
            nn.Linear(ff_hidden_size, hidden_size)
        )
        self.ff_norm = normalizations.LayerNorm32(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x, e, seq_lens, context, context_lens, **kwargs):
        """
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
        """
        assert e.dtype == torch.float32
        e = (self.modulation + e).chunk(6, dim=1)

        dtype = x.dtype

        # self-attention
        h = self.attn_norm(x).float() * (1 + e[1]) + e[0]
        y = self.attn(h.type(dtype), k_lens=seq_lens, **kwargs)
        x = x.float() + y.float() * e[2]

        y = self.de_attn(self.de_attn_norm(x).type(dtype), context, context, k_lens=context_lens)
        x = x + y

        h = self.ff_norm(x).float() * (1 + e[4]) + e[3]
        y = self.ff_proj(h.type(dtype))
        x = x.float() + y.float() * e[5]
        return x.type(dtype)


class WanCrossAttention2D(attentions.CrossAttention2D):
    def __init__(self, **kwargs):
        super().__init__(use_conv=False, **kwargs)
        self.to_qkv = nn.ModuleList([
            Linear(self.query_dim, self.model_dim, mode='ln', norm=normalizations.RMSNorm2D(self.model_dim)),
            Linear(self.context_dim, self.model_dim, mode='ln', norm=normalizations.RMSNorm2D(self.model_dim)),
            Linear(self.context_dim, self.model_dim, mode='l'),
        ])


class WanRotaryEmbedding(embeddings.RotaryEmbedding):
    def initialize_layers(self, dtype=torch.float64):
        # note, use fp64
        super().initialize_layers(dtype=dtype)

    def forward(self, x, grid_sizes=None, weights=None, **kwargs):
        dtype = x.dtype
        n, c = x.size(2), x.size(3) // 2

        # split freqs
        weights = weights.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # loop over samples
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w

            # precompute multipliers
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
            freqs_i = torch.cat([
                weights[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                weights[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                weights[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ], dim=-1).reshape(seq_len, 1, -1)

            # apply rotary embedding
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])

            # append to collection
            output.append(x_i)
        return torch.stack(output).type(dtype)  # otherwise fp64


@attentions.make_attend_fn.add_register()
class WanFlashAttend(nn.Module):
    half_dtypes = (torch.float16, torch.bfloat16)
    drop_prob = 0.0
    window_size = (-1, -1)

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.view_in = Rearrange('b n s d -> b s n d')
        self.view_out = Rearrange('b s n d -> b n s d')

    def force_to_half(self, x, dtype):
        return x if x.dtype in self.half_dtypes else x.to(dtype)

    def forward(
            self,
            q, k, v,
            q_lens=None, k_lens=None,
            softmax_scale=None, q_scale=None,
            causal=False, deterministic=False,
            force_dtype=torch.bfloat16,
            **kwargs
    ):
        """
        q:              [B, Lq, Nq, C1].
        k:              [B, Lk, Nk, C1].
        v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
        q_lens:         [B].
        k_lens:         [B].
        softmax_scale:  float. The scaling of QK^T before applying softmax.
        causal:         bool. Whether to apply causal attention mask.
        deterministic:  bool. If True, slightly slower and uses more memory.
        force_dtype:    torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
        """
        import flash_attn

        assert force_dtype in self.half_dtypes
        assert q.device.type == 'cuda' and q.size(-1) <= 256

        q, k, v = [self.view_in(x).contiguous() for x in (q, k, v)]

        # params
        b, lq, lk = q.size(0), q.size(1), k.size(1)
        out_dtype = q.dtype

        # preprocess query
        if q_lens is None:
            q = self.force_to_half(q.flatten(0, 1), force_dtype)
            q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
        else:
            q = self.force_to_half(torch.cat([u[:v] for u, v in zip(q, q_lens)]), force_dtype)

        # preprocess key, value
        if k_lens is None:
            k = self.force_to_half(k.flatten(0, 1), force_dtype)
            v = self.force_to_half(v.flatten(0, 1), force_dtype)
            k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
        else:
            k = self.force_to_half(torch.cat([u[:v] for u, v in zip(k, k_lens)]), force_dtype)
            v = self.force_to_half(torch.cat([u[:v] for u, v in zip(v, k_lens)]), force_dtype)

        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if q_scale is not None:
            q = q * q_scale

        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=self.drop_prob,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=self.window_size,
            deterministic=deterministic
        ).unflatten(0, (b, lq))

        x = self.view_out(x)
        return x.type(out_dtype)


class WanUNetHead(nn.Module):
    def __init__(self, in_features, out_features, patch_size, eps=1e-6):
        super().__init__()
        # layers
        out_features = math.prod(patch_size) * out_features
        self.norm = normalizations.LayerNorm32(in_features, elementwise_affine=False, eps=eps)
        self.head = nn.Linear(in_features, out_features)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, in_features) / in_features ** 0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        h = self.norm(x).float() * (1 + e[1]) + e[0]
        x = self.head(h.type(x.dtype))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_features, out_features, flf_pos_emb=False):
        super().__init__()
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(torch.zeros(1, 257 * 2, 1280))

        self.proj = torch.nn.Sequential(
            Linear(in_features, in_features, mode='nla', norm_fn=nn.LayerNorm, act=nn.GELU()),
            Linear(in_features, out_features, mode='ln', norm_fn=nn.LayerNorm)
        )

    def forward(self, image_embeds):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


@k_diffusion.make_schedule_fn.add_register()
class FlowUniPCMultistepSchedule(k_diffusion.Schedule):
    shift = 5.0
    sigma_min = 0

    def make_sigmas(self):
        s = 1 - 1 / self.timesteps
        sigma_max = self.shift * s / (1 + (self.shift - 1) * s)
        sigmas = torch.linspace(sigma_max, self.sigma_min, self.timesteps + 1)[:-1]

        sigma_last = torch.zeros(1)
        sigmas = torch.cat([sigmas, sigma_last]).flip([0])
        return sigmas
