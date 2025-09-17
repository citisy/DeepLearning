from functools import partial
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from utils import torch_utils
from .. import activations, attentions, bundles, embeddings, normalizations
from ..text_pretrain import llama, qwen2, transformers
from data_parse.nl_data_parse.pre_process.decoder import beam_search


class Config(bundles.Config):
    LayerNorm = 'LayerNorm'
    RMSNorm2D = 'RMSNorm2D'

    ScaleAttend = 'ScaleAttend'
    FlashAttend = 'FlashAttend'
    DynamicMemoryScaleAttend = 'DynamicMemoryScaleAttend'
    DynamicMemoryFlashAttend = 'DynamicMemoryFlashAttend'

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
            'model.layers.{0}.post_attention_layernorm': 'vlm.blocks.{0}.ff_res.norm',

            'lm_head': 'head'
        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        if 'head.weight' not in state_dict:
            state_dict['head.weight'] = state_dict['vlm.embed_tokens.weight']
        return state_dict


class Model(nn.Module):
    """https://github.com/QwenLM/Qwen2-VL"""
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    eos_ids = [151645, 151643]

    def __init__(self, vit_config=Config._2b_vit_config, vlm_config=Config._2b_vlm_config, model_config={}):  # noqa
        super().__init__()
        self.__dict__.update(model_config)

        self.vit = Vit(**vit_config)
        self.vlm = Vlm(**vlm_config)
        self.head = nn.Linear(self.vlm.hidden_size, self.vlm.vocab_size, bias=False)
        # note, officially use `vlm.embed_tokens.weight`!!!
        self.head.weight = self.vlm.embed_tokens.weight

        self.criterion = nn.CrossEntropyLoss()

        self.rope_deltas = None

    _device = None
    _dtype = None

    @property
    def device(self):
        return torch_utils.ModuleInfo.possible_device(self) if self._device is None else self._device

    @property
    def dtype(self):
        return torch_utils.ModuleInfo.possible_dtype(self) if self._dtype is None else self._dtype

    def set_half(self):
        dtype = torch.bfloat16

        torch_utils.ModuleManager.apply(
            self,
            lambda module: module.to(dtype),
            exclude=[normalizations.RMSNorm2D]
        )

        self.forward = partial(torch_utils.ModuleManager.assign_dtype_run, self, self.forward, dtype, force_effect_module=False)

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

    def make_caches(self):
        return [dict() for i in range(self.vlm.num_blocks)]

    def forward(self, *args, **kwargs):
        if self.training:
            return self.fit(*args, **kwargs)
        else:
            return self.inference(*args, **kwargs)

    def fit(self, text_ids, trues=None, **kwargs):
        preds = self.decode(text_ids)
        outputs = dict(
            preds=preds
        )
        outputs['loss'] = self.loss(trues, preds)
        return outputs

    def inference(
            self,
            text_ids, content_generator=True, seq_lens=None, vlm_past_kvs=None,
            **decode_kwargs
    ):
        if content_generator:
            if vlm_past_kvs is None:
                vlm_past_kvs = self.make_caches()

            preds = beam_search(text_ids, seq_lens, self.decode, eos_ids=self.eos_ids, vlm_past_kvs=vlm_past_kvs, **decode_kwargs)

            return dict(
                preds=preds,
                vlm_past_kvs=vlm_past_kvs
            )
        else:
            return self.decode(text_ids, **decode_kwargs)

    def decode(
            self,
            input_ids=None,
            image_pixel_values=None, image_grid_thw=None,
            video_pixel_values=None, video_grid_thw=None,
            start_pos=0, vlm_past_kvs=None,
            **kwargs
    ):
        inputs_embeds = self.make_input_embeds(
            input_ids,
            image_pixel_values=image_pixel_values if start_pos == 0 else None, image_grid_thw=image_grid_thw,
            video_pixel_values=video_pixel_values if start_pos == 0 else None, video_grid_thw=video_grid_thw,
        )

        # calculate RoPE index once per generation in the pre-fill stage only
        if start_pos == 0:
            attention_mask = torch.ones((1, inputs_embeds.shape[1])).to(inputs_embeds)
            position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
            self.rope_deltas = rope_deltas

        # then use the prev pre-calculated rope-deltas to get the correct position ids
        else:
            batch_size, seq_length, _ = inputs_embeds.shape

            delta = self.rope_deltas + start_pos
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)

            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        hidden_states = self.vlm(
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_kvs=vlm_past_kvs,
            start_pos=start_pos
        )
        hidden_states = self.head(hidden_states)
        return hidden_states

    def make_input_embeds(
            self, input_ids,
            image_pixel_values=None, image_grid_thw=None,
            video_pixel_values=None, video_grid_thw=None,
            **kwargs
    ):
        inputs_embeds = self.vlm.embed_tokens(input_ids)

        if image_pixel_values is not None:
            inputs_embeds = self.make_image_embeds(input_ids, image_pixel_values, image_grid_thw, inputs_embeds, **kwargs)

        if video_pixel_values is not None:
            inputs_embeds = self.make_video_embeds(input_ids, video_pixel_values, video_grid_thw, inputs_embeds, **kwargs)

        return inputs_embeds

    def make_image_embeds(self, input_ids, image_pixel_values, image_grid_thw, inputs_embeds, **kwargs):
        """for image inputs
        image_pixel_values: [b, w, h]
        image_grid_thw: [b, gw, gh]
        """
        image_embeds = self.vit(image_pixel_values, grid_thw=image_grid_thw, **kwargs)
        image_mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    def make_video_embeds(self, input_ids, video_pixel_values, video_grid_thw, inputs_embeds, **kwargs):
        """for video inputs"""
        video_embeds = self.vit(video_pixel_values, grid_thw=video_grid_thw, **kwargs)
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
        loss = self.criterion(shift_logits, shift_labels)
        return loss


def make_norm(*args, name=Config.LayerNorm, **kwargs):
    mapping = {
        Config.LayerNorm: nn.LayerNorm,
        Config.RMSNorm2D: normalizations.RMSNorm2D,
    }
    kwargs.setdefault('eps', 1e-6)
    return mapping[name](*args, **kwargs)


def make_base_attend_fn(name=Config.ScaleAttend):
    mapping = {
        Config.ScaleAttend: attentions.ScaleAttend,
        Config.FlashAttend: attentions.FlashAttend,
        Config.DynamicMemoryScaleAttend: partial(attentions.DynamicMemoryAttendWrapper, base_layer_fn=attentions.ScaleAttend),
        Config.DynamicMemoryFlashAttend: partial(attentions.DynamicMemoryAttendWrapper, base_layer_fn=attentions.FlashAttend),
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

        self.blocks = transformers.TransformerSequential(
            embed_dim, num_heads, int(embed_dim * mlp_ratio),
            attend_fn=attentions.RotaryAttendWrapper,
            attend_fn_kwargs=dict(
                embedding=self.rot_embedding,
                base_layer_fn=make_base_attend_fn(base_attend_name),
            ),
            fn_kwargs=dict(
                separate=False,
            ),
            ff_kwargs=dict(
                act=activations.FasterGELU(),
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
        self.num_blocks = num_blocks
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    _device = None
    _dtype = None

    @property
    def device(self):
        return torch_utils.ModuleInfo.possible_device(self) if self._device is None else self._device

    @property
    def dtype(self):
        return torch_utils.ModuleInfo.possible_dtype(self) if self._dtype is None else self._dtype

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
        hidden_states = self.patch_embed(hidden_states)

        attention_mask = self.make_attention_mask(hidden_states, grid_thw)
        rot_embedding_weights = self.rot_embedding.make_weights(grid_thw, self.spatial_merge_size).to(hidden_states.device)

        hidden_states = self.blocks(
            hidden_states, attention_mask=attention_mask,
            embedding_kwargs=dict(
                weights=rot_embedding_weights
            ),
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

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.fn = nn.Conv3d(in_ch, dim, kernel_size, stride=kernel_size, bias=bias)

    def forward(self, x):
        b = x.shape[0]
        # 'b s (c w h) -> k c tp p p'
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.fn(x)
        # 'k d tp p p -> b (k tp p p) d'
        x = x.view(b, -1, self.dim)
        return x


class RotaryEmbedding2D(embeddings.RotaryEmbedding):
    """for qk of rotary attention, not for seq"""

    def make_rot_pos_emb_kwargs(self, grid_thw, spatial_merge_size):
        position_ids = []
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
            position_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        position_ids = torch.cat(position_ids, dim=0)
        return position_ids

    def make_weights(self, grid_thw, spatial_merge_size):
        weights = super().make_weights(grid_thw[:, 1:].max())
        position_ids = self.make_rot_pos_emb_kwargs(grid_thw, spatial_merge_size)
        weights = weights[position_ids].flatten(1)
        weights = weights.repeat(1, 2)
        weights = weights[None, :, None, :]  # (s d) -> (1 s 1 d)
        return weights

    def forward(self, x, grid_thw=None, spatial_merge_size=None, weights=None):
        """x: (b s n d)"""
        if weights is None:
            weights = self.make_weights(grid_thw, spatial_merge_size)

        weights = weights.to(x.device)
        weights = torch.view_as_real(weights)
        cos = weights[..., 0]
        sin = weights[..., 1]
        x_ = x.float()
        y = (x_ * cos) + (self.rotate_half(x_) * sin)
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
            base_attend_name=Config.DynamicMemoryScaleAttend, norm_name=Config.RMSNorm2D,
            use_checkpoint=False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        self.rot_embedding = MRotaryEmbedding(hidden_size // num_heads, theta=1000000.0)

        self.blocks = transformers.TransformerSequential(
            hidden_size, num_heads, ff_hidden_size,
            attention_fn=qwen2.QwenSdpaAttention,
            fn_kwargs=dict(
                n_kv_heads=num_kv_heads,
            ),
            attend_fn=qwen2.QwenSdpaAttendWrapper,
            attend_fn_kwargs=dict(
                embedding=self.rot_embedding,
                base_layer_fn=make_base_attend_fn(base_attend_name),
            ),
            feed_forward_fn=llama.FeedForward,
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

    _device = None
    _dtype = None

    @property
    def device(self):
        return torch_utils.ModuleInfo.possible_device(self) if self._device is None else self._device

    @property
    def dtype(self):
        return torch_utils.ModuleInfo.possible_dtype(self) if self._dtype is None else self._dtype

    def forward(
            self,
            input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None,
            past_kvs=None, start_pos=0
    ):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = attentions.make_causal_attention_mask(inputs_embeds, start_pos=start_pos)

        rot_embedding_weights = self.rot_embedding.make_weights(position_ids=position_ids)

        hidden_states = inputs_embeds

        per_block_kwargs = []
        for past_kv in past_kvs:
            per_block_kwargs.append(dict(
                cache_fn=partial(attentions.DynamicMemoryAttendWrapper.cache, past_kv=past_kv),
            ))

        hidden_states = self.blocks(
            hidden_states, attention_mask=attention_mask,
            embedding_kwargs=dict(
                weights=rot_embedding_weights
            ),
            per_block_kwargs=per_block_kwargs
        )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class MRotaryEmbedding(RotaryEmbedding2D):
    """RotaryEmbedding3D"""
    mrope_section = [16, 24, 24] * 2

    def make_weights(self, position_ids):
        inv_freq_expanded = self.div_term[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        inv_freq_expanded = inv_freq_expanded.to(position_ids_expanded)
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

        x_ = x.float()
        y = (x_ * cos) + (self.rotate_half(x_) * sin)
        return y.type_as(x)
