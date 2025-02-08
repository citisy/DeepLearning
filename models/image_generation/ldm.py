import functools

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from utils import torch_utils
from . import VAE, ddpm, ddim, k_diffusion
from .ddpm import RandomOrLearnedSinusoidalPosEmb, ResnetBlock, make_norm_fn, append_dims
from .. import bundles, attentions
from ..attentions import CrossAttention2D, get_attention_input
from ..layers import Linear, Conv, Upsample, Downsample
from ..embeddings import SinusoidalEmbedding


class Config(ddim.Config):
    CROSSATTN = 0
    CROSSATTN_ADM = 1
    HYBRID = 2
    HYBRID_ADM = 3

    # for UNetModel label_emb num_classes
    # can not set to int value
    CONTINUOUS = 'continuous'
    TIMESTEP = 'timestep'
    SEQUENTIAL = 'sequential'

    # for sampler
    DDPM = 'ddpm'
    DDIM = 'ddim'
    EULER = 'Euler'

    model = dict()

    sampler = dict(
        name=DDIM,
        **ddim.Config.sampler,
    )

    backbone = dict(
        num_heads=8,
        attend_type='SplitScaleAttend'
    )
    vae = VAE.Config.get('backbone_32x32x4')

    default_model = 'vanilla'

    @classmethod
    def make_full_config(cls):
        config_dict = dict(
            vanilla=dict(
                model_config=cls.model,
                sampler_config=cls.sampler,
                backbone_config=cls.backbone,
                vae_config=cls.vae
            )
        )
        return config_dict


class WeightLoader(bundles.WeightLoader):
    @classmethod
    def auto_load(cls, save_path, save_name='', **kwargs):
        file_name = cls.get_file_name(save_path, save_name)
        state_dict = torch_utils.Load.from_file(file_name)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        return state_dict


class WeightConverter:
    cond_convert_dict = {}

    vae_convert_dict = {
        'first_stage_model': 'vae',
        **{'first_stage_model.' + k: 'vae.' + v for k, v in VAE.WeightConverter.convert_dict.items()}
    }

    backbone_convert_dict = {
        'model.diffusion_model': 'backbone',
        'model.diffusion_model.time_embed.0': 'backbone.time_embed.1.linear',
        'model.diffusion_model.time_embed.2': 'backbone.time_embed.2.linear',
        'model.diffusion_model.label_emb.0.0': 'backbone.label_emb.0',
        'model.diffusion_model.label_emb.0.2': 'backbone.label_emb.2',
        'model.diffusion_model.{2}.0.0': 'backbone.{2}.0.layers.0',
        'model.diffusion_model.{0}.0.in_layers.0': 'backbone.{0}.layers.0.in_layers.norm',
        'model.diffusion_model.{0}.0.in_layers.2': 'backbone.{0}.layers.0.in_layers.conv',
        'model.diffusion_model.{0}.0.emb_layers.1': 'backbone.{0}.layers.0.emb_layers.linear',
        'model.diffusion_model.{0}.0.out_layers.0': 'backbone.{0}.layers.0.norm',
        'model.diffusion_model.{0}.0.out_layers.3': 'backbone.{0}.layers.0.out_layers.conv',
        'model.diffusion_model.{0}.1.norm': 'backbone.{0}.layers.1.norm',
        'model.diffusion_model.{0}.1.proj_in': 'backbone.{0}.layers.1.proj_in',
        'model.diffusion_model.{0}.1.transformer_blocks.{2}.attn{1}.to_q': 'backbone.{0}.layers.1.transformer_blocks.{2}.attn{1}.to_qkv.0',
        'model.diffusion_model.{0}.1.transformer_blocks.{2}.attn{1}.to_k': 'backbone.{0}.layers.1.transformer_blocks.{2}.attn{1}.to_qkv.1',
        'model.diffusion_model.{0}.1.transformer_blocks.{2}.attn{1}.to_v': 'backbone.{0}.layers.1.transformer_blocks.{2}.attn{1}.to_qkv.2',
        'model.diffusion_model.{0}.1.transformer_blocks.{2}.attn{1}.to_out.0': 'backbone.{0}.layers.1.transformer_blocks.{2}.attn{1}.to_out.linear',
        'model.diffusion_model.{0}.1.transformer_blocks.{2}.ff.net.0.proj': 'backbone.{0}.layers.1.transformer_blocks.{2}.ff.0.proj',
        'model.diffusion_model.{0}.1.transformer_blocks.{2}.ff.net.2': 'backbone.{0}.layers.1.transformer_blocks.{2}.ff.2',
        'model.diffusion_model.{0}.1.transformer_blocks.{2}.norm{1}.': 'backbone.{0}.layers.1.transformer_blocks.{2}.norm{1}.',
        'model.diffusion_model.{0}.1.proj_out': 'backbone.{0}.layers.1.proj_out',
        'model.diffusion_model.{0}.1.conv': 'backbone.{0}.layers.1.op.1',
        'model.diffusion_model.{0}.2.conv': 'backbone.{0}.layers.2.op.1',
        'model.diffusion_model.{0}.0.op': 'backbone.{0}.layers.0.op',
        'model.diffusion_model.{0}.0.skip_connection': 'backbone.{0}.layers.0.proj',
        'model.diffusion_model.middle_block.2.in_layers.0': 'backbone.middle_block.layers.2.in_layers.norm',
        'model.diffusion_model.middle_block.2.in_layers.2': 'backbone.middle_block.layers.2.in_layers.conv',
        'model.diffusion_model.middle_block.2.out_layers.0': 'backbone.middle_block.layers.2.norm',
        'model.diffusion_model.middle_block.2.emb_layers.1': 'backbone.middle_block.layers.2.emb_layers.linear',
        'model.diffusion_model.middle_block.2.out_layers.3': 'backbone.middle_block.layers.2.out_layers.conv',
        'model.diffusion_model.out.0': 'backbone.out.norm',
        'model.diffusion_model.out.2': 'backbone.out.conv',
    }

    transpose_keys = ()

    @classmethod
    def from_official(cls, state_dict):
        """convert weights from official model to my own model
        see https://github.com/CompVis/latent-diffusion?tab=readme-ov-file#model-zoo
        to get more detail

        Usage:
            .. code-block:: python

                state_dict = torch.load(self.pretrain_model, map_location=self.device)['state_dict']
                state_dict = WeightConverter.from_official(state_dict)
                Model(...).load_state_dict(state_dict)
        """
        convert_dict = {
            **cls.vae_convert_dict,
            **cls.backbone_convert_dict,
            **cls.cond_convert_dict,
        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        for k in cls.transpose_keys:
            state_dict[k] = state_dict[k].t()

        return state_dict

    @classmethod
    def to_official(cls, state_dict):
        pass

    @classmethod
    def from_official_lora(cls, state_dict):
        """see https://github.com/kohya-ss/sd-scripts"""
        cond_convert_dict = {}
        for k, v in cls.cond_convert_dict.items():
            k = '.'.join(k.split('.')[2:])
            k = ('lora_te.' + k).replace('.', '_')
            cond_convert_dict[k] = v

        backbone_convert_dict = {}
        for k, v in cls.backbone_convert_dict.items():
            k = k.replace('model.diffusion_model.', '')

            # note, usually value
            num_layers = 12

            if '{0}.1' in k:
                _k = k.replace('{0}.1', f"mid_block.attentions.0")
                _k = ('lora_unet.' + _k).replace('.', '_')
                backbone_convert_dict[_k] = v.replace('{0}', 'middle_block')

                for i in range(num_layers):
                    if i > 0:
                        # from https://github.com/kohya-ss/sd-scripts/blob/main/library/model_util.py#L299
                        block_id = (i - 1) // (2 + 1)
                        layer_in_block_id = (i - 1) % (2 + 1)
                        _k = k.replace('{0}.1', f"down_blocks.{block_id}.attentions.{layer_in_block_id}")
                        _k = ('lora_unet.' + _k).replace('.', '_')
                        backbone_convert_dict[_k] = v.replace('{0}', f"input_blocks.{i}")

                    # from https://github.com/kohya-ss/sd-scripts/blob/main/library/model_util.py#L335
                    block_id = i // (2 + 1)
                    layer_in_block_id = i % (2 + 1)
                    _k = k.replace('{0}.1', f"up_blocks.{block_id}.attentions.{layer_in_block_id}")
                    _k = ('lora_unet.' + _k).replace('.', '_')
                    backbone_convert_dict[_k] = v.replace('{0}', f"output_blocks.{i}")

            else:
                k = ('lora_unet.' + k).replace('.', '_')
                backbone_convert_dict[k] = v

        convert_dict = {
            **cond_convert_dict,
            **backbone_convert_dict,
        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        convert_dict = {
            '{0}.lora_down': '{0}.down',
            '{0}.lora_up': '{0}.up',
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        return state_dict

    @classmethod
    def to_official_lora(cls):
        pass

    @classmethod
    def from_official_controlnet(cls, state_dict):
        """see https://github.com/lllyasviel/ControlNet?tab=readme-ov-file"""
        convert_dict = {}
        for k, v in cls.backbone_convert_dict.items():
            convert_dict[k.replace('model.diffusion_model', 'control_model')] = v.replace('backbone', 'backbone.control_model')

        convert_dict.update({
            'control_model.zero_convs.{0}.0': 'backbone.control_model.zero_convs.{0}.layers.0',
            'control_model.input_hint_block.{[0]}.': 'backbone.control_model.input_hint_block.layers.{([0]+1)//2}.conv.',
            'control_model.middle_block_out': 'backbone.control_model.middle_block_out.layers'
        })

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


class Model(ddim.Model):
    """refer to:
    paper:
        - High-Resolution Image Synthesis with Latent Diffusion Models
    code:
        - https://github.com/CompVis/latent-diffusion
        - https://github.com/CompVis/stable-diffusion
    """

    scale_factor = 0.18215
    cond_trainable = True
    vae_trainable = False
    inference_only = False

    sampler_mapping = {
        Config.DDPM: ddpm.Sampler,
        Config.DDIM: ddim.Sampler,
        Config.EULER: k_diffusion.EulerSampler
    }

    def set_low_memory_run(self):
        super().set_low_memory_run()
        self.make_txt_cond = functools.partial(torch_utils.ModuleManager.low_memory_run, self.cond, self.make_txt_cond, self.device)

    def make_sampler(self, sampler_config=Config.sampler, **kwargs):
        self.sampler = self.sampler_mapping.get(sampler_config['name'], sampler_config['name'])(**sampler_config)

    def make_cond(self, **kwargs):
        raise NotImplementedError

    def make_diffuse(self, vae_config=Config.vae, backbone_config=Config.backbone, **kwargs):
        cond = self.make_cond(**kwargs)
        vae = VAE.Model(self.img_ch, **vae_config)  # decode is in module, encode is head module
        backbone = UNetModel(vae.z_ch, cond.output_size, **backbone_config)

        if not hasattr(cond, 'encode'):
            cond.encode = cond.__call__
        assert hasattr(vae, 'decode')
        assert hasattr(vae, 'encode')

        if not self.cond_trainable:
            torch_utils.ModuleManager.freeze_module(cond)
        if not self.vae_trainable:
            torch_utils.ModuleManager.freeze_module(vae)

        self.cond = cond
        self.backbone = backbone
        self.vae = vae

    @property
    def diffuse_in_ch(self):
        return self.vae.z_ch

    @property
    def diffuse_in_size(self):
        return (
            self.image_size[0] // self.vae.encoder.down_scale,
            self.image_size[1] // self.vae.encoder.down_scale,
        )

    def loss(self, x, text_ids=None, **kwargs):
        txt_cond = self.make_txt_cond(text_ids, **kwargs)
        kwargs.update(txt_cond)

        z, _, _ = self.vae.encode(x)
        x0 = self.scale_factor * z

        return self.sampler.loss(self.diffuse, x0, **kwargs)

    def post_process(self, x=None, text_ids=None, mask_x=None, **kwargs):
        """

        Args:
            x (torch.Tensor): (b, c, h, w)
                original images, if given, run the img2img mode
            text_ids (torch.Tensor): (b, l)
                positive prompt
            neg_text_ids (torch.Tensor): (b, l)
            text_weights (torch.Tensor): (b, l)
            neg_text_weights (torch.Tensor): (b, l)
            mask_x (torch.Tensor): (b, 1, h, w)
                mask images, if given, run the inpaint mode
                suggest has the same (h, w) of x
                fall in [0, 1], 0 gives masked
            scale:
            strength:

        Returns:
            images
        """
        b = len(text_ids)

        txt_cond = self.make_txt_cond(text_ids, **kwargs)
        kwargs.update(txt_cond)

        # make x_t
        if x is None or not len(x):  # txt2img
            x = self.gen_x_t(b)
            x = self.sampler.scale_x_t(x)
            z0 = None

        else:  # img2img
            x, z0, i0 = self.make_image_cond(x, **kwargs)
            kwargs.update(i0=i0)

        z = self.sampler(self.diffuse, x, **kwargs)
        z = z / self.scale_factor

        if x is not None and len(x) and mask_x is not None and len(mask_x):
            # todo: apply for different conditioning_key
            mask_x = F.interpolate(mask_x, size=z.shape[-2:])
            z = z0 * mask_x + z * (1 - mask_x)

        images = self.vae.decode(z)

        return images

    def make_txt_cond(self, text_ids, neg_text_ids=None, text_weights=None, neg_text_weights=None, scale=7.5, **kwargs) -> dict:
        c = self.cond.encode(text_ids)
        uc = None
        if scale > 1.0 and neg_text_ids is not None:
            uc = self.cond.encode(neg_text_ids)

        if text_weights is not None:
            c = self.cond_with_weights(c, text_weights)

        if neg_text_ids is not None and neg_text_weights is not None:
            uc = self.cond_with_weights(uc, neg_text_weights)

        return dict(
            cond=c,
            un_cond=uc
        )

    def cond_with_weights(self, c, weights):
        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        original_mean = c.mean()
        c = c * append_dims(weights, len(c.shape)).expand(c.shape)
        new_mean = c.mean()
        c = c * (original_mean / new_mean)
        return c

    def make_image_cond(self, images, i0=None, noise=None, strength=0.75, **kwargs):
        z, _, _ = self.vae.encode(images)
        x0 = self.scale_factor * z

        if i0 is None:
            i0 = int(strength * self.sampler.num_steps)

        timestep_seq = self.sampler.make_timesteps(i0)
        t = timestep_seq[0]
        t = torch.full((x0.shape[0],), t, device=x0.device, dtype=torch.long)
        xt = self.sampler.q_sample(x0, t, noise=noise)
        return xt, z, i0

    def diffuse(self, x, time, cond=None, un_cond=None, scale=7.5, **backbone_kwargs):
        if un_cond is not None:
            x = torch.cat([x] * 2)
            time = torch.cat([time] * 2)
            cond = torch.cat([un_cond, cond])

        z = self.backbone(x, timesteps=time, context=cond, **backbone_kwargs)
        if un_cond is not None:
            e_t_uncond, e_t = z.chunk(2)
            e_t = e_t_uncond + scale * (e_t - e_t_uncond)
        else:
            e_t = z

        return e_t


class Txt2ImgModel4Triton(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, neg_token):
        """

        Args:
            tokens: torch.int64, (b, seq_len)
            neg_token: torch.int64, (b, seq_len)

        Returns:
            torch.uint8, (b, h, w, c)
        """
        fake_x = self.model(x=None, text_ids=tokens, neg_text_ids=neg_token)
        fake_x = (fake_x + 1) * 0.5  # unnormalize, [-1, 1] -> [0, 1]
        fake_x = fake_x.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).contiguous().to(torch.uint8)
        return fake_x


class Img2ImgModel4Triton(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, neg_token, images):
        """

        Args:
            tokens: torch.int64, (b, seq_len)
            neg_token: torch.int64, (b, seq_len)
            images: torch.uint8, (b, h, w, c)

        Returns:
            torch.uint8, (b, h, w, c)
        """
        images = images / 255.
        images = images * 2 - 1  # normalize, [0, 1] -> [-1, 1]
        fake_x = self.model(x=images, text_ids=tokens, neg_text_ids=neg_token)
        fake_x = (fake_x + 1) * 0.5  # unnormalize, [-1, 1] -> [0, 1]
        fake_x = fake_x.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).contiguous().to(torch.uint8)
        return fake_x


class UNetModel(nn.Module):
    """base on Unet, add attention, res, etc."""

    def __init__(
            self,
            in_ch, context_dim,
            out_ch=4, unit_dim=320, ch_mult=(1, 2, 4, 4),  # for model
            use_checkpoint=True, attend_type='ScaleAttend',  # for resnet and transformers
            sinusoidal_pos_emb_theta=10000, learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16,  # for time embed
            num_classes=None, adm_in_channels=None,  # for label_emb
            groups=32, num_res_blocks=2,  # for resnet
            num_heads=None, head_dim=None, transformer_depth=1, use_linear_in_transformer=False, attend_layers=(0, 1, 2),  # for transformers
            conv_resample=True,  # for up/down sample
    ):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.context_dim = context_dim
        self.num_classes = num_classes

        time_emb_dim = unit_dim * 4

        if isinstance(transformer_depth, int):
            transformer_depth = len(ch_mult) * [transformer_depth]

        # helper
        make_res = functools.partial(ResnetBlock, groups=groups, use_checkpoint=use_checkpoint, time_emb_dim=time_emb_dim)
        make_trans = functools.partial(TransformerBlock, context_dim=context_dim, use_checkpoint=use_checkpoint, use_linear=use_linear_in_transformer, attend_type=attend_type)

        if learned_sinusoidal_cond:
            sin_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sin_pos_emb = SinusoidalEmbedding(unit_dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = unit_dim

        self.time_embed = nn.Sequential(
            sin_pos_emb,
            Linear(fourier_dim, time_emb_dim, mode='la', act=nn.SiLU()),
            Linear(time_emb_dim, time_emb_dim, mode='l'),
        )

        if num_classes is not None:
            self.label_emb = self.make_label_emb(num_classes, time_emb_dim, unit_dim, adm_in_channels)

        out_ch = unit_dim
        layers = [TimestepEmbedSequential(nn.Conv2d(in_ch, out_ch, 3, padding=1))]
        input_block_chans = [out_ch]

        num_stages = len(ch_mult)
        in_ch = out_ch
        for i, mult in enumerate(ch_mult):
            is_bottom = i == num_stages - 1

            for _ in range(num_res_blocks):
                out_ch = mult * unit_dim
                blocks = [make_res(in_ch, out_ch)]
                if i in attend_layers:
                    _n_heads, _, _head_dim = get_attention_input(num_heads, out_ch, head_dim)
                    blocks.append(make_trans(out_ch, _n_heads, _head_dim, depth=transformer_depth[i]))

                layers.append(TimestepEmbedSequential(*blocks))
                input_block_chans.append(out_ch)
                in_ch = out_ch
            if not is_bottom:
                layers.append(TimestepEmbedSequential(
                    Downsample(out_ch, out_ch, use_conv=conv_resample)
                ))
                input_block_chans.append(out_ch)

        self.input_blocks = nn.ModuleList(layers)

        _n_heads, _, _head_dim = get_attention_input(num_heads, out_ch, head_dim)
        self.middle_block = TimestepEmbedSequential(
            make_res(out_ch, out_ch),
            make_trans(out_ch, _n_heads, _head_dim, depth=transformer_depth[-1]),
            make_res(out_ch, out_ch),
        )

        layers = []
        for i, mult in list(enumerate(ch_mult))[::-1]:
            is_top = i == 0
            for j in range(num_res_blocks + 1):
                is_block_bottom = j == num_res_blocks

                ich = input_block_chans.pop()
                out_ch = unit_dim * mult
                blocks = [make_res(in_ch + ich, out_ch)]
                if i in attend_layers:
                    _n_heads, _, _head_dim = get_attention_input(num_heads, out_ch, head_dim)
                    blocks.append(make_trans(out_ch, _n_heads, _head_dim, depth=transformer_depth[i]))

                if not is_top and is_block_bottom:
                    blocks.append(Upsample(out_ch, out_ch, use_conv=conv_resample))

                layers.append(TimestepEmbedSequential(*blocks))
                in_ch = out_ch

        self.output_blocks = nn.ModuleList(layers)
        self.out = Conv(in_ch, self.out_channels, 3, mode='nac', act=nn.SiLU(), norm=make_norm_fn(groups, unit_dim))

    def make_label_emb(self, num_classes, time_emb_dim, unit_dim, adm_in_channels):
        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                label_emb = nn.Embedding(num_classes, time_emb_dim)
            elif self.num_classes == Config.CONTINUOUS:
                label_emb = nn.Linear(1, time_emb_dim)
            elif self.num_classes == Config.TIMESTEP:
                label_emb = nn.Sequential(
                    SinusoidalEmbedding(unit_dim),
                    nn.Sequential(
                        nn.Linear(unit_dim, time_emb_dim),
                        nn.SiLU(),
                        nn.Linear(time_emb_dim, time_emb_dim),
                    ),
                )
            elif self.num_classes == Config.SEQUENTIAL:
                assert adm_in_channels is not None
                label_emb = nn.Sequential(
                    nn.Linear(adm_in_channels, time_emb_dim),
                    nn.SiLU(),
                    nn.Linear(time_emb_dim, time_emb_dim),
                )
            else:
                raise ValueError

            return label_emb

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        emb = self.time_embed(timesteps.to(x))

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0], f'Expect {y.shape[0]}, but got {x.shape[0]}'
            emb = emb + self.label_emb(y)

        h = x
        hs = []
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        return self.out(h)


class TimestepEmbedSequential(nn.Module):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, emb=None, context=None):
        for layer in self.layers:
            if isinstance(layer, ResnetBlock):
                x = layer(x, emb)
            elif isinstance(layer, TransformerBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, in_ch, n_heads, head_dim, groups=32,
                 depth=1, dropout=0., context_dim=None, use_linear=False, use_checkpoint=False, attend_type='ScaleAttend'):
        super().__init__()
        self.in_channels = in_ch
        self.use_linear = use_linear

        model_dim = n_heads * head_dim

        if not isinstance(context_dim, list):
            context_dim = [context_dim] * depth

        assert len(context_dim) == depth

        self.norm = make_norm_fn(groups, in_ch, eps=1e-6)  # note, original code use `eps=1e-6`

        if use_linear:
            self.proj_in = nn.Linear(in_ch, model_dim)
        else:
            self.proj_in = nn.Conv2d(in_ch, model_dim, 1)

        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(
            model_dim, n_heads, head_dim, drop_prob=dropout, context_dim=d, attend_type=attend_type
        ) for d in context_dim])

        if use_linear:
            self.proj_out = nn.Linear(model_dim, in_ch)
        else:
            self.proj_out = nn.Conv2d(model_dim, in_ch, 1, stride=1, padding=0)

        if use_checkpoint:
            self.forward = functools.partial(torch_utils.ModuleManager.checkpoint, self, self.forward, is_first_layer=True)

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape

        y = self.norm(x)
        if self.use_linear:
            y = rearrange(y, 'b c h w -> b (h w) c')
            y = self.proj_in(y)
        else:
            y = self.proj_in(y)
            y = rearrange(y, 'b c h w -> b (h w) c')

        for block in self.transformer_blocks:
            y = block(y, context=context)

        if self.use_linear:
            y = self.proj_out(y)
            y = rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)
        else:
            y = rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)
            y = self.proj_out(y)
        return y + x


class BasicTransformerBlock(nn.Module):
    def __init__(self, query_dim, n_heads, head_dim, attend_type='ScaleAttend', drop_prob=0., context_dim=None, gated_ff=True):
        super().__init__()
        attend_fn = attentions.make_attend_fn.get(attend_type)
        self.norm1 = nn.LayerNorm(query_dim)
        self.attn1 = CrossAttention2D(query_dim=query_dim, n_heads=n_heads, head_dim=head_dim, attend=attend_fn(), drop_prob=drop_prob, bias=False)  # is a self-attention
        self.attn1.to_out.linear.bias = nn.Parameter(torch.empty(query_dim))  # only to_out module has bias

        self.norm2 = nn.LayerNorm(query_dim)
        self.attn2 = CrossAttention2D(query_dim=query_dim, context_dim=context_dim, n_heads=n_heads, head_dim=head_dim, attend=attend_fn(), drop_prob=drop_prob, bias=False)  # is self-attn if context is none
        self.attn2.to_out.linear.bias = nn.Parameter(torch.empty(query_dim))  # only to_out module has bias

        self.norm3 = nn.LayerNorm(query_dim)
        self.ff = FeedForward(query_dim, drop_prob=drop_prob, glu=gated_ff)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class FeedForward(nn.Sequential):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, drop_prob=0.):
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        super().__init__(
            project_in,
            nn.Dropout(drop_prob),
            nn.Linear(inner_dim, dim_out)
        )


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
