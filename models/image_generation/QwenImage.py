from collections import OrderedDict
from functools import partial

import torch
from torch import Tensor, nn

from utils import torch_utils
from ..multimodal_pretrain import Qwen2_5_VL
from .. import attentions, bundles, normalizations, embeddings
from . import flux, k_diffusion, QwenVAE
from .. import layers
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange


class Config(bundles.Config):
    backbone = dict(
        in_ch=64,
        out_ch=64,
        context_in_dim=3584,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        num_blocks=60,
        axes_dim=[16, 56, 56],
        separate=True,
        head_mode=1
    )

    text_encoder = dict(
        **Qwen2_5_VL.Config.get('7b'),
        model_config=dict(
            share_head=False
        )
    )

    default_model = ''

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            '': dict(
                text_encoder_config=cls.text_encoder,
                backbone_config=cls.backbone,
                vae_config=QwenVAE.Config.vae,
                sampler_config=flux.Config.sampler,
            ),
        }


class WeightConverter:
    diffusers_backbone_convert_dict = {
        'txt_in': 'txt_in.linear',
        'txt_norm': 'txt_in.norm',
        'time_text_embed.timestep_embedder.linear_{0}.': 'time_in.{[0]-1}.linear.',
        'norm_out.linear': 'head.adaLN_modulation.linear',
        'proj_out': 'head.linear',

        'transformer_blocks.{0}.attn.add_q_proj': 'transformer_blocks.{0}.txt_stream.to_qkv.0',
        'transformer_blocks.{0}.attn.add_k_proj': 'transformer_blocks.{0}.txt_stream.to_qkv.1',
        'transformer_blocks.{0}.attn.add_v_proj': 'transformer_blocks.{0}.txt_stream.to_qkv.2',
        'transformer_blocks.{0}.attn.norm_added_k': 'transformer_blocks.{0}.txt_stream.key_norm',
        'transformer_blocks.{0}.attn.norm_added_q': 'transformer_blocks.{0}.txt_stream.query_norm',
        'transformer_blocks.{0}.attn.to_add_out': 'transformer_blocks.{0}.txt_stream.proj',
        'transformer_blocks.{0}.txt_mlp.net.0.proj': 'transformer_blocks.{0}.txt_stream.mlp.0.linear',
        'transformer_blocks.{0}.txt_mlp.net.2': 'transformer_blocks.{0}.txt_stream.mlp.1.linear',
        'transformer_blocks.{0}.txt_mod.1': 'transformer_blocks.{0}.txt_stream.mod.lin',

        'transformer_blocks.{0}.attn.norm_k': 'transformer_blocks.{0}.img_stream.key_norm',
        'transformer_blocks.{0}.attn.norm_q': 'transformer_blocks.{0}.img_stream.query_norm',
        'transformer_blocks.{0}.attn.to_out.0': 'transformer_blocks.{0}.img_stream.proj',
        'transformer_blocks.{0}.attn.to_q': 'transformer_blocks.{0}.img_stream.to_qkv.0',
        'transformer_blocks.{0}.attn.to_k': 'transformer_blocks.{0}.img_stream.to_qkv.1',
        'transformer_blocks.{0}.attn.to_v': 'transformer_blocks.{0}.img_stream.to_qkv.2',
        'transformer_blocks.{0}.img_mlp.net.0.proj': 'transformer_blocks.{0}.img_stream.mlp.0.linear',
        'transformer_blocks.{0}.img_mlp.net.2': 'transformer_blocks.{0}.img_stream.mlp.1.linear',
        'transformer_blocks.{0}.img_mod.1': 'transformer_blocks.{0}.img_stream.mod.lin',
    }

    @classmethod
    def from_diffusers(cls, state_dicts):
        """weights trained from `diffusers`
        Args:
            state_dicts:
                {
                    "text_encoder": tensors,
                    "backbone": tensors,
                    "vae": tensors,
                }

        """
        state_dict = OrderedDict()

        if 'backbone' in state_dicts:
            _state_dict = torch_utils.Converter.convert_keys(state_dicts['backbone'], cls.diffusers_backbone_convert_dict)
            state_dict.update({'backbone.' + k: v for k, v in _state_dict.items()})

        if 'text_encoder' in state_dicts:
            _state_dict = Qwen2_5_VL.WeightConverter.from_official(state_dicts['text_encoder'])
            state_dict.update({'text_encoder.' + k: v for k, v in _state_dict.items()})

        if 'vae' in state_dicts:
            _state_dict = torch_utils.Converter.convert_keys(state_dicts['vae'], QwenVAE.WeightConverter.convert_dict)
            state_dict.update({'vae.' + k.replace('gamma', 'weight'): v for k, v in _state_dict.items()})

        return state_dict


class Model(flux.Model):
    """https://github.com/QwenLM/Qwen-Image"""

    def __init__(
            self,
            text_encoder_config=Config.text_encoder,
            backbone_config=Config.backbone,
            vae_config=QwenVAE.Config.vae,
            sampler_config=flux.Config.sampler,
            model_config=dict(),
            **kwargs
    ):
        super(flux.Model, self).__init__()
        self.__dict__.update(model_config)

        self.text_encoder = Qwen2_5_VL.Model(**text_encoder_config)
        self.text_encoder.encode = self.text_encoder.__call__
        self.callback = Callback(self.text_encoder.vlm.num_blocks - 1)

        self.backbone = QwenImage(**backbone_config)
        self.vae = Vae(**vae_config)
        self.sampler = QwenSampler(**sampler_config)

        self.set_module_status()

    vae_trainable = False
    text_encoder_trainable = False
    backbone_trainable = False

    def set_module_status(self):
        if not self.text_encoder_trainable:
            torch_utils.ModuleManager.freeze_module(self.text_encoder, only_submodules=True)

        if not self.vae_trainable:
            torch_utils.ModuleManager.freeze_module(self.vae, only_submodules=True)
            self.vae.set_inference_only()

        if not self.backbone_trainable:
            torch_utils.ModuleManager.freeze_module(self.backbone, only_submodules=True)

    def set_low_memory_run(self):
        # Not critical to run single batch for decoding strategy, but reduce more GPU memory
        self.vae.encode = partial(torch_utils.ModuleManager.single_batch_run, self.vae, self.vae.encode)
        self.vae.decode = partial(torch_utils.ModuleManager.single_batch_run, self.vae, self.vae.decode)

        def wrap1(module, func, **kwargs1):
            # note, device would be changed after model initialization.
            def wrap2(*args, **kwargs2):
                return torch_utils.ModuleManager.low_memory_run(module, func, self.device, *args, **kwargs1, **kwargs2)

            return wrap2

        self.text_encoder.encode = wrap1(self.text_encoder, self.text_encoder.encode)
        self.vae.encode = wrap1(self.vae, self.vae.encode)
        self.vae.decode = wrap1(self.vae, self.vae.decode)
        self.sampler.forward = wrap1(self.backbone, self.sampler.forward)
        self.sampler.loss = wrap1(self.backbone, self.sampler.loss)
        self.sampler.to(self.device)

    def set_half(self):
        dtype = torch.bfloat16

        torch_utils.ModuleManager.apply(
            self,
            lambda module: module.to(dtype),
            include=['text_encoder', 'backbone', 'vae'],
            exclude=[normalizations.GroupNorm32, normalizations.RMSNorm, embeddings.SinusoidalEmbedding]
        )

        def wrap1(module, func):
            def wrap2(*args, **kwargs):
                return torch_utils.ModuleManager.assign_dtype_run(module, func, dtype, *args, force_effect_module=False, **kwargs)

            return wrap2

        self.text_encoder.encode = wrap1(self.text_encoder, self.text_encoder.encode)
        self.vae.encode = wrap1(self.vae, self.vae.encode)
        self.vae.decode = wrap1(self.vae, self.vae.decode)
        self.sampler.forward = wrap1(self.backbone, self.sampler.forward)
        self.sampler.loss = wrap1(self.backbone, self.sampler.loss)
        self.sampler.to(self.dtype)

    def forward(self, **kwargs):
        if self.training:
            raise NotImplementedError
        else:
            return self.inference(**kwargs)

    def inference(
            self, x=None, mask_x=None, image_size=None,
            text_ids=None, text_conds_attention_mask=None, template_seq_lens=None, text_conds=None,
            neg_text_ids=None, neg_text_conds_attention_mask=None, neg_template_seq_lens=None, neg_text_conds=None,
            **kwargs
    ):
        if text_conds is None:
            text_conds, text_conds_attention_mask = self.make_text_cond(text_ids, template_seq_lens)

        if neg_text_conds is None:
            neg_text_conds, neg_text_conds_attention_mask = self.make_text_cond(neg_text_ids, neg_template_seq_lens)

        # pad_seq = lambda x, max_seq_len: torch.cat([x, x.new_zeros(x.shape[0], max_seq_len - x.shape[1], x.shape[-1])], dim=1)
        # pad_mask = lambda x, max_seq_len: torch.cat([x, x.new_zeros(x.shape[0], max_seq_len - x.shape[1])], dim=1)
        #
        # max_seq_len = max(text_conds.shape[1], neg_text_conds.shape[1])
        # text_conds = pad_seq(text_conds, max_seq_len)
        # text_conds_attention_mask = pad_mask(text_conds_attention_mask, max_seq_len)
        # neg_text_conds = pad_seq(neg_text_conds, max_seq_len)
        # neg_text_conds_attention_mask = pad_mask(neg_text_conds_attention_mask, max_seq_len)

        if x is None or not len(x):  # txt2img
            x = self.gen_x_t(text_conds.shape[0], image_size)
            z0 = None
            bs, c, h, w = x.shape
            sigmas = self.sampler.schedule.make_sigmas(mu=self.sampler.schedule.make_mu(image_seq_len=h * w // 4))

        else:  # img2img
            x, z0, i0, sigmas = self.make_image_cond(x, noise=self.gen_x_t(text_conds.shape[0], (x.shape[-1], x.shape[-2])), **kwargs)
            kwargs.update(i0=i0)

        kwargs.update(
            text_conds=text_conds,
            text_conds_attention_mask=text_conds_attention_mask,
            neg_text_conds=neg_text_conds,
            neg_text_conds_attention_mask=neg_text_conds_attention_mask,
        )

        z = self.sampler(self.process, x, sigmas=sigmas, **kwargs)

        if x is not None and len(x) and mask_x is not None and len(mask_x):
            # todo: apply for different conditioning_key
            mask_x = F.interpolate(mask_x, size=z.shape[-2:])
            z = z0 * mask_x + z * (1 - mask_x)

        images = self.vae.decode(z)[:, :, 0]

        return images

    def make_text_cond(self, text_ids, template_seq_lens):
        self.text_encoder.encode(text_ids, callback_fn=self.callback)
        text_conds = self.callback.cache_hidden_state
        split_hidden_states = [text_cond[template_seq_len:] for text_cond, template_seq_len in zip(text_conds, template_seq_lens)]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([u.shape[0] for u in split_hidden_states])
        text_conds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        text_conds_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        return text_conds, text_conds_attention_mask

    # def process(
    #         self, img, t_vec, img_cond=None,
    #         text_conds=None, text_conds_attention_mask=None,
    #         neg_text_conds=None, neg_text_conds_attention_mask=None,
    #         scale=4.0,
    #         **kwargs
    # ):
    #     """flow process"""
    #     bs, c, H, W = img.shape
    #     h = H // 2
    #     w = W // 2
    #
    #     img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    #     t_vec = torch.full((img.shape[0],), t_vec[0], dtype=img.dtype, device=img.device)
    #
    #     if neg_text_conds is not None:
    #         img = torch.repeat_interleave(img, 2, dim=0)
    #         t_vec = torch.repeat_interleave(t_vec, 2, dim=0)
    #         text_conds = torch.cat([text_conds, neg_text_conds])
    #         text_conds_attention_mask = torch.cat([text_conds_attention_mask, neg_text_conds_attention_mask])
    #
    #     img_shapes = [(1, h, w)] * bs
    #
    #     e_t = self.backbone(
    #         img=torch.cat((img, img_cond), dim=-1) if img_cond is not None else img,
    #         img_shapes=img_shapes,
    #         txt=text_conds,
    #         text_conds_attention_mask=text_conds_attention_mask,
    #         timesteps=t_vec,
    #     )
    #     e_t = rearrange(e_t, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=2, pw=2)
    #
    #     if neg_text_conds is not None:
    #         e_t, e_t_uncond = e_t.chunk(2)
    #         comb_e_t = e_t_uncond + scale * (e_t - e_t_uncond)
    #         cond_norm = torch.norm(e_t, dim=-1, keepdim=True)
    #         noise_norm = torch.norm(comb_e_t, dim=-1, keepdim=True)
    #         e_t = comb_e_t * (cond_norm / noise_norm)
    #     return e_t

    def process(
            self, img, t_vec, img_cond=None,
            text_conds=None, text_conds_attention_mask=None,
            neg_text_conds=None, neg_text_conds_attention_mask=None,
            scale=4.0,
            **kwargs
    ):
        """flow process"""
        bs, c, H, W = img.shape
        h = H // 2
        w = W // 2

        img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        t_vec = torch.full((img.shape[0],), t_vec[0], dtype=img.dtype, device=img.device)

        img_shapes = [(1, h, w)] * bs

        e_t = self.backbone(
            img=torch.cat((img, img_cond), dim=-1) if img_cond is not None else img,
            img_shapes=img_shapes,
            txt=text_conds,
            text_conds_attention_mask=text_conds_attention_mask,
            timesteps=t_vec,
        )

        if neg_text_conds is not None:
            e_t_uncond = self.backbone(
                img=torch.cat((img, img_cond), dim=-1) if img_cond is not None else img,
                img_shapes=img_shapes,
                txt=neg_text_conds,
                text_conds_attention_mask=neg_text_conds_attention_mask,
                timesteps=t_vec,
            )

            comb_e_t = e_t_uncond + scale * (e_t - e_t_uncond)
            cond_norm = torch.norm(e_t, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_e_t, dim=-1, keepdim=True)
            e_t = comb_e_t * (cond_norm / noise_norm)
        e_t = rearrange(e_t, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h, w=w, ph=2, pw=2)
        return e_t


class QwenImage(nn.Module):
    def __init__(
            self,
            in_ch, out_ch,
            hidden_size, num_heads,
            context_in_dim, mlp_ratio,
            num_blocks,
            separate=False, head_mode=0,
            axes_dim=(16, 56, 56),
            use_checkpoint=True, **kwargs
    ):
        super().__init__()

        self.in_channels = in_ch
        self.out_channels = out_ch
        self.use_checkpoint = use_checkpoint

        self.img_in = nn.Linear(in_ch, hidden_size)
        self.txt_in = layers.Linear(context_in_dim, hidden_size, mode='nl', norm=normalizations.RMSNorm2D(context_in_dim))

        self.time_embed = embeddings.SinusoidalEmbedding(256)
        self.time_in = flux.MLPEmbedder(256, hidden_size)

        self.embedding = QwenRotaryEmbedding(axes_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attend=attentions.FlashAttend(),
                    embedding=self.embedding,
                    separate=separate
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = flux.Head(hidden_size, 1, out_ch, head_mode=head_mode)
        if use_checkpoint:
            self.forward = partial(torch_utils.ModuleManager.checkpoint, self, self.forward)

    def forward(self, img, timesteps, img_shapes, txt, text_conds_attention_mask=None):
        # running on sequences img
        img = self.img_in(img)
        txt = self.txt_in(txt)
        vec = self.time_in(self.time_embed(timesteps))

        pe = self.embedding.make_weights(img_shapes, txt.shape[1])

        img, txt = self.forward_transformer_blocks(img, txt, vec, pe)
        img = self.head(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    def forward_transformer_blocks(self, img, txt, vec, pe):
        """easy to wrap"""
        for block in self.transformer_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        return img, txt


class Callback(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.cache_hidden_state = None

    def forward(self, i, h):
        if i == self.layer_idx:
            self.cache_hidden_state = h


class QwenTimestepProjEmbedding(nn.Module):
    def __init__(self, embedding_dim, use_additional_t_cond=False):
        super().__init__()

        self.time_proj = embeddings.SinusoidalEmbedding(256)
        self.timestep_embedder = flux.MLPEmbedder(256, embedding_dim)
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, embedding_dim)

    def forward(self, timestep, hidden_states, addition_t_cond=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_states.dtype))  # (N, D)

        conditioning = timesteps_emb
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                raise ValueError("When additional_t_cond is True, addition_t_cond must be provided.")
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            addition_t_emb = addition_t_emb.to(dtype=hidden_states.dtype)
            conditioning = conditioning + addition_t_emb

        return conditioning


class QwenSampler(flux.FlowMatchEulerSampler):
    def p_sample(self, diffuse_func, x_t, t, prev_t=None, num_steps=None, sigmas=None, **diffuse_kwargs):
        # todo: add more sample methods
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_t, device=x_t.device, dtype=torch.long)

        sigma = k_diffusion.extract(sigmas, t, x_t.shape)
        next_sigma = k_diffusion.extract(sigmas, prev_t, x_t.shape)

        gamma = torch.where(
            torch.logical_and(self.s_tmin <= sigma, sigma <= self.s_tmax),
            min(self.s_churn / (num_steps - 1 + 1e-8), 2 ** 0.5 - 1),
            0.
        ).to(sigma)

        sigma_hat = sigma * (gamma + 1.0)

        if torch.any(gamma > 0):
            eps = torch.randn_like(x_t) * self.s_noise
            x_t = x_t + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5

        possible_sigma = sigmas[self.sigma_to_idx(sigma_hat, sigmas)]
        c_skip, c_out, c_in = self.scaling(possible_sigma)
        c_skip, c_out, c_in = [k_diffusion.append_dims(c, len(x_t.shape)) for c in (c_skip, c_out, c_in)]
        possible_t = self.make_p_sample_possible_t(possible_sigma, sigmas)

        d = diffuse_func(c_in * x_t, possible_t, **diffuse_kwargs)
        dt = next_sigma - sigma_hat

        x_t = x_t + d * dt
        return x_t, None


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio, attend=None, embedding=None, separate=False):
        super().__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_stream = flux.CondStreamBlock(hidden_size, num_heads, mlp_ratio, double=True, separate=separate)
        self.txt_stream = flux.CondStreamBlock(hidden_size, num_heads, mlp_ratio, double=True, separate=separate)
        self.attend = attend
        self.embedding = embedding
        self.view_in = Rearrange('b h s d -> b s h d')
        self.view_out = Rearrange('b h s d -> b s (h d)')

    def forward(self, img, txt, vec, pe) -> tuple[Tensor, Tensor]:
        txt_mod1, txt_mod2, txt_q, txt_k, txt_v, _ = self.txt_stream.stream_in(txt, vec)
        img_mod1, img_mod2, img_q, img_k, img_v, _ = self.img_stream.stream_in(img, vec)

        img_freqs, txt_freqs = pe

        txt_q, txt_k, txt_v, img_q, img_k, img_v = [self.view_in(x).contiguous() for x in (txt_q, txt_k, txt_v, img_q, img_k, img_v)]

        txt_q = self.embedding(txt_q, txt_freqs)
        txt_k = self.embedding(txt_k, txt_freqs)
        img_q = self.embedding(img_q, img_freqs)
        img_k = self.embedding(img_k, img_freqs)

        txt_q, txt_k, txt_v, img_q, img_k, img_v = [self.view_in(x).contiguous() for x in (txt_q, txt_k, txt_v, img_q, img_k, img_v)]
        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = self.attend(q, k, v)
        attn = self.view_out(attn)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        txt = self.txt_stream.stream_out(txt, txt_attn, txt_mod1, txt_mod2)
        img = self.img_stream.stream_out(img, img_attn, img_mod1, img_mod2)

        return img, txt


class QwenRotaryEmbedding(nn.Module):
    def __init__(self, embedding_dims: list[int], theta=10000, scale_rope=True):
        super().__init__()
        self.theta = theta
        self.embedding_dims = embedding_dims
        self.scale_rope = scale_rope
        self.initialize_layers()

    def _apply(self, fn, recurse=True):
        """apply for meta load"""
        if self.pos_freqs.is_meta:
            self.initialize_layers()
        return super()._apply(fn, recurse)

    def initialize_layers(self):
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.embedding_dims[0], self.theta),
                self.rope_params(pos_index, self.embedding_dims[1], self.theta),
                self.rope_params(pos_index, self.embedding_dims[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.embedding_dims[0], self.theta),
                self.rope_params(neg_index, self.embedding_dims[1], self.theta),
                self.rope_params(neg_index, self.embedding_dims[2], self.theta),
            ],
            dim=1,
        )

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def make_weights(self, video_fhw, max_txt_seq_len) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            video_fhw (`tuple[int, int, int]` or `list[tuple[int, int, int]]`):
                A list of 3 integers [frame, height, width] representing the shape of the video.
            max_txt_seq_len (`int` or `torch.Tensor`, *optional*):
                The maximum text sequence length for RoPE computation. This should match the encoder hidden states
                sequence length. Can be either an int or a scalar tensor (for torch.compile compatibility).
        """
        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            # RoPE frequencies are cached via a lru_cache decorator on _compute_video_freqs
            video_freq = self._compute_video_freqs(frame, height, width, idx)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_txt_seq_len_int = int(max_txt_seq_len)
        # Use cached device-transferred freqs to avoid CPU→GPU sync every forward call
        txt_freqs = self.pos_freqs[max_vid_index: max_vid_index + max_txt_seq_len_int, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    def _compute_video_freqs(
            self, frame: int, height: int, width: int, idx: int = 0
    ) -> torch.Tensor:
        seq_lens = frame * height * width
        pos_freqs, neg_freqs = self.pos_freqs, self.neg_freqs

        freqs_pos = pos_freqs.split([x // 2 for x in self.embedding_dims], dim=1)
        freqs_neg = neg_freqs.split([x // 2 for x in self.embedding_dims], dim=1)

        freqs_frame = freqs_pos[0][idx: idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2):], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2):], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()

    def forward(self, x, freqs_cis):
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1).to(x.device)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class Vae(QwenVAE.Model):
    spatial_compression_ratio = 2 ** 3

    # The minimal tile height and width for spatial tiling to be used
    tile_sample_min_height = 256
    tile_sample_min_width = 256

    # The minimal distance between two spatial tiles
    tile_sample_stride_height = 192
    tile_sample_stride_width = 192

    def decode(self, z):
        scale_factor = self.scale_factor
        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.to(z)
        shift_factor = self.shift_factor
        if isinstance(shift_factor, torch.Tensor):
            shift_factor = shift_factor.to(z)

        z = z / scale_factor + shift_factor

        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                time = []
                feat_cache = [None] * self.count_conv3d(self.decoder)
                for k in range(num_frames):
                    tile = z[:, :, k: k + 1, i: i + tile_latent_min_height, j: j + tile_latent_min_width]
                    tile = self.post_quant_conv(tile)
                    feat_idx = [0]
                    decoded = self.decoder(tile, feat_cache=feat_cache, feat_idx=feat_idx)
                    time.append(decoded)
                row.append(torch.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
        return dec

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                    y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                    x / blend_extent
            )
        return b
