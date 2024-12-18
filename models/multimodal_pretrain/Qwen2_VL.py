from functools import partial
from typing import Optional

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F

from utils import torch_utils
from .. import bundles
from ..activations import FasterGELU
from .. import attentions
from ..attentions import RotaryAttendWrapper, get_attention_input, get_qkv
from ..embeddings import RotaryEmbedding
from ..layers import Linear
from ..normalizations import RMSNorm2D
from ..text_pretrain.llama import FeedForward
from ..text_pretrain.transformers import TransformerSequential, make_causal_attention_mask


class Config(bundles.Config):
    LayerNorm = 'LayerNorm'
    RMSNorm2D = 'RMSNorm2D'

    ScaleAttend = 'ScaleAttend'
    FlashAttend = 'FlashAttend'

    _2b_vit_config = dict(
        output_size=1536
    )

    _2b_vlm_config = dict(
        hidden_size=1536,
        ff_hidden_size=8960,
        num_heads=12,
        num_kv_heads=2,
        vocab_size=151936
    )

    _7b_vit_config = dict(
        output_size=3584
    )

    _7b_vlm_config = dict(
        hidden_size=3584,
        ff_hidden_size=18944,
        num_heads=28,
        num_kv_heads=4,
        vocab_size=152064
    )

    _72b_vit_config = dict(
        output_size=8192
    )

    _72b_vlm_config = dict(
        hidden_size=8192,
        ff_hidden_size=29568,
        num_blocks=80,
        num_heads=64,
        num_kv_heads=8,
        vocab_size=152064
    )

    default_model = '2b'

    @classmethod
    def make_full_config(cls):
        return {
            '2b': dict(
                vit_config=cls._2b_vit_config,
                vlm_config=cls._2b_vlm_config
            ),

            '7b': dict(
                vit_config=cls._7b_vit_config,
                vlm_config=cls._7b_vlm_config
            ),

            '72b': dict(
                vit_config=cls._72b_vit_config,
                vlm_config=cls._72b_vlm_config
            )
        }


class WeightLoader(bundles.WeightLoader):
    pass


class WeightConverter:
    @classmethod
    def from_official(cls, state_dict):
        convert_dict = {
            'visual': 'vit',
            'visual.patch_embed.proj': 'vit.patch_embed.fn',
            'visual.blocks.{0}.norm1': 'vit.blocks.{0}.attn_res.norm',
            'visual.blocks.{0}.norm2': 'vit.blocks.{0}.ff_res.norm',
            'visual.blocks.{0}.attn.qkv': 'vit.blocks.{0}.attn_res.fn.to_qkv',
            'visual.blocks.{0}.attn.proj': 'vit.blocks.{0}.attn_res.fn.to_out.linear',
            'visual.blocks.{0}.mlp.fc1': 'vit.blocks.{0}.ff_res.fn.0.linear',
            'visual.blocks.{0}.mlp.act': 'vit.blocks.{0}.ff_res.fn.0.act',
            'visual.blocks.{0}.mlp.fc2': 'vit.blocks.{0}.ff_res.fn.1.linear',

            'model': 'vlm',
            'model.layers.{0}.self_attn.q_proj': 'vlm.blocks.{0}.attn_res.fn.to_qkv.0',
            'model.layers.{0}.self_attn.k_proj': 'vlm.blocks.{0}.attn_res.fn.to_qkv.1',
            'model.layers.{0}.self_attn.v_proj': 'vlm.blocks.{0}.attn_res.fn.to_qkv.2',
            'model.layers.{0}.self_attn.o_proj': 'vlm.blocks.{0}.attn_res.fn.to_out.linear',
            'model.layers.{0}.mlp.gate_proj': 'vlm.blocks.{0}.ff_res.fn.f1.linear',
            'model.layers.{0}.mlp.up_proj': 'vlm.blocks.{0}.ff_res.fn.f3.linear',
            'model.layers.{0}.mlp.down_proj': 'vlm.blocks.{0}.ff_res.fn.f2.linear',
            'model.layers.{0}.input_layernorm': 'vlm.blocks.{0}.attn_res.norm',
            'model.layers.{0}.post_attention_layernorm': 'vlm.blocks.{0}.ff_res.norm'
        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


class Model(nn.Module):
    """https://github.com/QwenLM/Qwen2-VL"""
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652

    def __init__(self, vit_config=Config._2b_vit_config, vlm_config=Config._2b_vlm_config, model_config={}):
        super().__init__()
        self.__dict__.update(model_config)

        self.vit = Vit(**vit_config)
        self.vlm = Vlm(**vlm_config)
        self.head = nn.Linear(self.vlm.hidden_size, self.vlm.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

        self.rope_deltas = None

    def get_rope_index(
            self,
            input_ids: torch.LongTensor,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        """copy from `transformers`
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.vit.spatial_merge_size
        image_token_id = self.image_token_id
        video_token_id = self.video_token_id
        vision_start_token_id = self.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def forward(
            self,
            input_ids,
            attention_mask=None, position_ids=None, cache_position=None,
            pixel_values=None, image_grid_thw=None,
            pixel_values_videos=None, video_grid_thw=None,
            labels=None
    ):
        inputs_embeds = self.vlm.embed_tokens(input_ids)
        if pixel_values is not None:
            inputs_embeds = self.make_image_embeds(input_ids, pixel_values, image_grid_thw, inputs_embeds)

        if pixel_values_videos is not None:
            inputs_embeds = self.make_video_embeds(input_ids, pixel_values_videos, video_grid_thw, inputs_embeds)

        if position_ids is None and input_ids is not None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (cache_position is not None and cache_position[0] == 0) or self.rope_deltas is None:
                position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        hidden_states = self.vlm(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
        )
        outputs = dict(
            hidden_states=hidden_states
        )

        if self.training:
            logits = self.head(hidden_states)
            outputs['loss'] = self.loss(labels, logits)

        return outputs

    def make_image_embeds(self, input_ids, pixel_values, image_grid_thw, inputs_embeds):
        """for image inputs"""
        image_embeds = self.vit(pixel_values, grid_thw=image_grid_thw)
        image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    def make_video_embeds(self, input_ids, pixel_values_videos, video_grid_thw, inputs_embeds):
        """for video inputs"""
        video_embeds = self.vit(pixel_values_videos, grid_thw=video_grid_thw)
        video_mask = (input_ids == self.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        return inputs_embeds

    def loss(self, trues, preds):
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        preds = preds.float()
        # Shift so that tokens < n predict n
        shift_logits = preds[..., :-1, :].contiguous()
        shift_labels = trues[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        loss = self.loss_fn(shift_logits, shift_labels)
        return loss


def make_norm(*args, name=Config.LayerNorm, **kwargs):
    mapping = {
        Config.LayerNorm: nn.LayerNorm,
        Config.RMSNorm2D: RMSNorm2D,
    }
    kwargs.setdefault('eps', 1e-6)
    return mapping[name](*args, **kwargs)


def make_base_attend_fn(name=Config.ScaleAttend):
    mapping = {
        Config.ScaleAttend: attentions.ScaleAttend,
        Config.FlashAttend: attentions.FlashAttend
    }
    return mapping[name]


class Vit(nn.Module):
    """Vision Transformer"""

    def __init__(
            self,
            in_ch=3, embed_dim=1280, output_size=1536,
            patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
            num_heads=16, mlp_ratio=4, num_blocks=32,
            base_attend_name=Config.ScaleAttend,
            use_checkpoint=False,
            **kwargs
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding3D(
            in_ch=in_ch,
            dim=embed_dim,
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
        )

        self.rot_embedding = RotaryEmbedding2D(embed_dim // num_heads // 2)

        self.blocks = TransformerSequential(
            embed_dim, num_heads, int(embed_dim * mlp_ratio),
            attend_fn=RotaryAttendWrapper,
            attend_fn_kwargs=dict(
                embedding=self.rot_embedding,
                base_layer_fn=make_base_attend_fn(base_attend_name),
            ),
            fn_kwargs=dict(
                separate=False,
            ),
            ff_kwargs=dict(
                act=FasterGELU(),
            ),
            norm_fn=make_norm,
            norm_first=True,

            num_blocks=num_blocks,
            use_checkpoint=use_checkpoint
        )
        self.merger = PatchMerger(
            output_size=output_size, input_size=embed_dim, spatial_merge_size=spatial_merge_size
        )

        self.spatial_merge_size = spatial_merge_size

    def make_attention_mask(self, hidden_states, grid_thw):
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        seq_length = hidden_states.shape[1]
        attention_mask = torch.zeros([1, 1, seq_length, seq_length], device=hidden_states.device, dtype=torch.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = True

        return attention_mask

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)[None]

        attention_mask = self.make_attention_mask(hidden_states, grid_thw)
        # max_grid_size, position_ids = self.make_rot_pos_emb_kwargs(grid_thw)
        max_grid_size, position_ids = self.rot_embedding.make_rot_pos_emb_kwargs(grid_thw, self.spatial_merge_size)
        rot_embedding_weights = self.rot_embedding.make_weights(hidden_states.shape[1])

        hidden_states = self.blocks(
            hidden_states, attention_mask=attention_mask,
            embedding_kwargs=dict(
                grid_size=max_grid_size,
                position_ids=position_ids,
                weights=rot_embedding_weights
            )
        )

        return self.merger(hidden_states)


class PatchEmbedding3D(nn.Module):
    """for video patch embedding"""

    def __init__(self, dim, patch_size, in_ch=3, temporal_patch_size=2, bias=False):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.in_channels = in_ch
        self.temporal_patch_size = temporal_patch_size

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.fn = nn.Conv3d(in_ch, dim, kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x):
        # 's (c w h) -> k c tp p p'
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.fn(x)
        # 'k d tp p p -> (k tp p p) d'
        x = x.view(-1, self.dim)
        return x


class RotaryEmbedding2D(RotaryEmbedding):
    """for qk of rotary attention, not for seq"""

    def make_rot_pos_emb_kwargs(self, grid_thw, spatial_merge_size):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        return max_grid_size, pos_ids

    def forward(self, x, grid_size, position_ids, weights=None):
        """x: (b s n d)"""
        if weights is None:
            weights = self.make_weights(x.shape[1])

        weights = weights.to(x.device)
        weights = weights.repeat(1, 2)
        weights = weights[:grid_size]
        weights = weights[position_ids].flatten(1)
        weights = weights[None, :, None, :]  # (s d) -> (1 s 1 d)
        weights = torch.view_as_real(weights)
        cos = weights[..., 0]
        sin = weights[..., 1]
        x = x.float()
        y = (x * cos) + (self.rotate_half(x) * sin)
        return y.type_as(x)

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class PatchMerger(nn.Module):
    def __init__(self, output_size: int, input_size: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = input_size * (spatial_merge_size ** 2)
        self.ln_q = make_norm(input_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x


class Vlm(nn.Module):
    """Vision Language Model"""

    def __init__(
            self,
            pad_token_id=None, vocab_size=151936,
            hidden_size=1536, ff_hidden_size=8960,
            num_heads=12, num_kv_heads=2, num_blocks=28,
            base_attend_name=Config.ScaleAttend, norm_name=Config.RMSNorm2D,
            use_checkpoint=False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        self.rot_embedding = MRotaryEmbedding(hidden_size // num_heads, theta=1000000.0)

        self.blocks = TransformerSequential(
            hidden_size, num_heads, ff_hidden_size,
            attention_fn=VisionSdpaAttention,
            fn_kwargs=dict(
                n_kv_heads=num_kv_heads,
            ),
            attend_fn=VisionSdpaAttend,
            attend_fn_kwargs=dict(
                embedding=self.rot_embedding,
                base_layer_fn=make_base_attend_fn(base_attend_name),
            ),
            feed_forward_fn=FeedForward,
            ff_kwargs=dict(
                bias=False,
            ),
            norm_fn=make_norm,
            norm_kwargs=dict(
                name=norm_name
            ),
            norm_first=True,

            num_blocks=num_blocks,
            use_checkpoint=use_checkpoint

        )
        self.norm = make_norm(hidden_size, name=norm_name)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None, past_key_values=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)

        if attention_mask is None:
            attention_mask = make_causal_attention_mask(inputs_embeds)
        rot_embedding_weights = self.rot_embedding.make_weights(position_ids=position_ids)

        hidden_states = inputs_embeds

        hidden_states = self.blocks(
            hidden_states, attention_mask=attention_mask,
            embedding_kwargs=dict(
                weights=rot_embedding_weights
            )
        )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class VisionSdpaAttention(nn.Module):
    """cross attention"""

    def __init__(self, n_heads=None, model_dim=None, head_dim=None, n_kv_heads=None,
                 attend=None, out_layer=None, **fn_kwargs):
        super().__init__()
        n_heads, model_dim, head_dim = get_attention_input(n_heads, model_dim, head_dim)
        query_dim = model_dim
        context_dim = n_kv_heads * head_dim

        # note, mainly differences, [model_dim, ...] not [..., model_dim]
        self.to_qkv = nn.ModuleList([
            nn.Linear(model_dim, query_dim, bias=True, **fn_kwargs),
            nn.Linear(model_dim, context_dim, bias=True, **fn_kwargs),
            nn.Linear(model_dim, context_dim, bias=True, **fn_kwargs),
        ])

        self.q_view_in = Rearrange('b s (n dk)-> b n s dk', n=n_heads)
        self.kv_view_in = Rearrange('b s (n dk)-> b n s dk', n=n_kv_heads)

        self.view_out = Rearrange('b n s dk -> b s (n dk)')
        self.to_out = Linear(model_dim, query_dim, mode='l', bias=False, **fn_kwargs) if out_layer is None else out_layer

        self.attend = VisionSdpaAttend() if attend is None else attend

    def forward(self, q, k=None, v=None, attention_mask=None, **attend_kwargs):
        q, k, v = get_qkv(q, k, v)
        q, k, v = [m(x) for m, x in zip(self.to_qkv, (q, k, v))]

        q = self.q_view_in(q).contiguous()
        k = self.kv_view_in(k).contiguous()
        v = self.kv_view_in(v).contiguous()

        x = self.attend(q, k, v, attention_mask=attention_mask, **attend_kwargs)
        x = self.view_out(x)

        x = self.to_out(x)
        return x


class VisionSdpaAttend(RotaryAttendWrapper):
    def forward(self, q, k, v, attention_mask=None, embedding_kwargs=dict(), **attend_kwargs):
        """
        in(q|k|v): (b n s d)
        out(attn): (b n s d)
        """
        q, k, v = [self.view_in(x).contiguous() for x in (q, k, v)]
        q = self.embedding(q, **embedding_kwargs)
        k = self.embedding(k, **embedding_kwargs)
        q, k, v = [self.view_out(x).contiguous() for x in (q, k, v)]
        ratio = q.shape[1] // k.shape[1]
        # note, mainly difference.
        k = k.repeat_interleave(ratio, dim=1)
        v = v.repeat_interleave(ratio, dim=1)
        attn = self.base_layer(q, k, v, attention_mask=attention_mask, **attend_kwargs)
        return attn


class MRotaryEmbedding(RotaryEmbedding2D):
    """RotaryEmbedding3D"""
    mrope_section = [16, 24, 24] * 2

    def make_weights(self, position_ids):
        inv_freq_expanded = self.div_term[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))], dim=-1)[:, :, None, :]
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))], dim=-1)[:, :, None, :]

        return cos, sin

    def forward(self, x, position_ids=None, weights=None):
        if weights is None:
            weights = self.make_weights(position_ids)

        cos, sin = weights
        cos = cos.to(x.device)
        sin = sin.to(x.device)

        y = (x * cos) + (self.rotate_half(x) * sin)
        return y
