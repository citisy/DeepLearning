from functools import partial

import torch
from einops.layers.torch import Rearrange
from torch import nn

from utils import torch_utils
from .. import attentions, bundles, embeddings, normalizations
from ..layers import Linear
from ..text_pretrain import transformers, llama, MoE  # register ff_fn


class Config(bundles.Config):
    _0_5b_decoder = dict(
        hidden_size=896,
        ff_hidden_size=4864,
        num_heads=14,
        num_blocks=24,
        num_kv_heads=2,
        vocab_size=151936
    )

    _1_5b_decoder = dict(
        hidden_size=1536,
        ff_hidden_size=8960,
        num_heads=12,
        num_blocks=28,
        num_kv_heads=2,
        vocab_size=151936
    )

    _7b_decoder = dict(
        hidden_size=3584,
        ff_hidden_size=18944,
        num_heads=28,
        num_blocks=28,
        num_kv_heads=4,
        vocab_size=152064
    )

    _72b_decoder = dict(
        hidden_size=8192,
        ff_hidden_size=29568,
        num_heads=64,
        num_blocks=80,
        num_kv_heads=8,
        vocab_size=152064
    )

    _7b_moe_decoder = dict(
        **_7b_decoder,
        ff_type='MoeFeedForward',
        ff_kwargs=dict(
            bias=False,
            experts_hidden_size=1408,
            share_experts_hidden_size=5632,
            num_experts=60,
            top_k=4
        )
    )

    default_model = '7b'

    @classmethod
    def make_full_config(cls):
        return {
            '0.5b': dict(
                decoder_config=cls._0_5b_decoder
            ),

            '1.5b': dict(
                decoder_config=cls._1_5b_decoder
            ),

            '7b': dict(
                decoder_config=cls._7b_decoder,
                model_config=dict(
                    share_head=False
                )
            ),

            '72b': dict(
                decoder_config=cls._72b_decoder,
                model_config=dict(
                    share_head=False
                )
            ),

            '7b_moe': dict(
                decoder_config=cls._7b_moe_decoder,
                model_config=dict(
                    share_head=False,
                    use_moe=True
                )
            ),
        }


class WeightLoader(bundles.WeightLoader):
    pass


class WeightConverter(bundles.WeightConverter):
    moe_convert_dict = {
        'model.layers.{0}.mlp.{1}.gate_proj': 'decoder.blocks.{0}.ff_res.fn.{1}.f1.linear',
        'model.layers.{0}.mlp.{1}.up_proj': 'decoder.blocks.{0}.ff_res.fn.{1}.f3.linear',
        'model.layers.{0}.mlp.{1}.down_proj': 'decoder.blocks.{0}.ff_res.fn.{1}.f2.linear',
        'model.layers.{0}.mlp.gate': 'decoder.blocks.{0}.ff_res.fn.gate',
        'model.layers.{0}.mlp.shared_expert_gate': 'decoder.blocks.{0}.ff_res.fn.shared_expert_gate'
    }

    convert_dict = {
        'model.layers.{0}.self_attn.q_proj': 'decoder.blocks.{0}.attn_res.fn.to_qkv.0',
        'model.layers.{0}.self_attn.k_proj': 'decoder.blocks.{0}.attn_res.fn.to_qkv.1',
        'model.layers.{0}.self_attn.v_proj': 'decoder.blocks.{0}.attn_res.fn.to_qkv.2',
        'model.layers.{0}.self_attn.o_proj': 'decoder.blocks.{0}.attn_res.fn.to_out.linear',
        'model.layers.{0}.mlp.gate_proj': 'decoder.blocks.{0}.ff_res.fn.f1.linear',
        'model.layers.{0}.mlp.up_proj': 'decoder.blocks.{0}.ff_res.fn.f3.linear',
        'model.layers.{0}.mlp.down_proj': 'decoder.blocks.{0}.ff_res.fn.f2.linear',
        'model.layers.{0}.input_layernorm': 'decoder.blocks.{0}.attn_res.norm',
        'model.layers.{0}.post_attention_layernorm': 'decoder.blocks.{0}.ff_res.norm',

        **moe_convert_dict,

        'model.embed_tokens': 'embedding',
        'model.norm': 'decoder.norm',
        'lm_head': 'head'
    }


class Model(nn.Module):
    """https://github.com/QwenLM/Qwen2"""
    pad_id = 151643
    ignore_id = -100
    eos_ids = [151645, 151643]
    share_head = True
    use_moe = False

    def __init__(self, decoder_config=Config._7b_decoder, model_config={}):
        super().__init__()
        self.__dict__.update(model_config)

        self.decoder = Decoder(**decoder_config)
        self.embedding = nn.Embedding(self.decoder.vocab_size, self.decoder.hidden_size, self.pad_id)
        if self.share_head:
            self.head = lambda x: x.matmul(self.embedding.weight.transpose(1, 0))
        else:
            self.head = nn.Linear(self.decoder.hidden_size, self.decoder.vocab_size, bias=False)

        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_id)
        if self.use_moe:
            self.moe_criterion = MoE.MoeLoss(**decoder_config['ff_kwargs'])

    def forward(self, *args, **kwargs):
        if self.training:
            return self.fit(*args, **kwargs)
        else:
            return self.inference(*args, **kwargs)

    def fit(self, text_ids, label_ids, seq_lens=None, **decode_kwargs):
        ff_caches = [dict() for _ in range(self.decoder.num_blocks)] if self.use_moe else None
        logits = self.decode(text_ids, ff_caches=ff_caches, **decode_kwargs)
        return dict(
            logits=logits,
            **self.loss(logits, label_ids, ff_caches, seq_lens),
        )

    def loss(self, logits, label_ids, ff_caches=None, seq_lens=None):
        logits = logits.float()
        loss = self.criterion(
            logits.view(-1, logits.size(-1)),
            label_ids.contiguous().view(-1)
        )
        out = dict(
            loss=loss,
        )
        if self.use_moe:
            pad_mask = attentions.make_pad_mask(seq_lens, max_len=label_ids.shape[1])
            moe_loss = self.moe_criterion(ff_caches, pad_mask)
            out['loss'] = loss + moe_loss
            out['loss.moe'] = moe_loss
        return out

    def inference(self, text_ids, **decode_kwargs):
        logits = self.decode(text_ids, **decode_kwargs)
        return dict(
            logits=logits,
        )

    def decode(self, x, start_pos=0, attention_mask=None, **decoder_kwargs):
        x = self.embedding(x)
        if attention_mask is None:
            attention_mask = attentions.make_causal_attention_mask(x, start_pos=start_pos)
        x = self.decoder(x, attention_mask=attention_mask, start_pos=start_pos, **decoder_kwargs)
        x = self.head(x)
        return x


class Decoder(nn.Module):
    def __init__(
            self,
            vocab_size=151936,
            hidden_size=3584, ff_hidden_size=18944,
            num_heads=28, num_blocks=28, num_kv_heads=4,
            rot_theta=1000000.0,
            attend_type='FlashAttendWithDynamicMemory',
            ff_type='GateFeedForward', ff_kwargs=dict(bias=False),
            use_checkpoint=False
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads

        self.rot_embedding = RotaryEmbedding(hidden_size // num_heads, theta=rot_theta)

        self.blocks = transformers.TransformerSequential(
            hidden_size, num_heads, ff_hidden_size,
            attention_fn=QwenSdpaAttention,
            fn_kwargs=dict(
                n_kv_heads=num_kv_heads,
            ),
            attend_fn=QwenSdpaAttendWrapper,
            attend_fn_kwargs=dict(
                embedding=self.rot_embedding,
                base_layer_fn=attentions.make_attend_fn.get(attend_type)
            ),
            feed_forward_fn=transformers.make_ff_fn.get(ff_type),
            ff_kwargs=ff_kwargs,
            norm_fn=normalizations.RMSNorm2D,
            norm_first=True,

            num_blocks=num_blocks,
            use_checkpoint=use_checkpoint
        )
        self.norm = normalizations.RMSNorm2D(hidden_size)

    # _device = None
    # _dtype = None
    #
    # @property
    # def device(self):
    #     return torch_utils.ModuleInfo.possible_device(self) if self._device is None else self._device
    #
    # @property
    # def dtype(self):
    #     return torch_utils.ModuleInfo.possible_dtype(self) if self._dtype is None else self._dtype

    def make_caches(self):
        return [dict() for _ in range(self.num_blocks)]

    def forward(
            self,
            inputs_embeds=None, attention_mask=None,
            past_kvs=None, start_pos=0, ff_caches=None,
            **ignore_kwargs
    ):
        if attention_mask is None:
            attention_mask = attentions.make_causal_attention_mask(inputs_embeds, start_pos=start_pos)

        hidden_states = inputs_embeds

        per_block_kwargs = [dict() for _ in range(self.num_blocks)]
        if past_kvs:
            for i, past_kv in enumerate(past_kvs):
                per_block_kwargs[i]['cache_fn'] = partial(attentions.DynamicMemoryAttendWrapper.cache, past_kv=past_kv)

        if ff_caches:
            for i, ff_cache in enumerate(ff_caches):
                per_block_kwargs[i]['ff_kwargs'] = dict(ff_cache=ff_cache)

        seq_len = inputs_embeds.shape[1]
        if past_kvs and 'k' in past_kvs[0]:
            seq_len += past_kvs[0]['k'].shape[-2]
        rot_embedding_weights = self.rot_embedding.make_weights(seq_len)

        hidden_states = self.blocks(
            hidden_states, attention_mask=attention_mask,
            embedding_kwargs=dict(
                weights=rot_embedding_weights,
                start_pos=start_pos
            ),
            per_block_kwargs=per_block_kwargs,
        )

        hidden_states = self.norm(hidden_states)

        return hidden_states


class RotaryEmbedding(embeddings.RotaryEmbedding):
    def make_weights(self, seq_len):
        device = self.div_term.device
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            position_ids = torch.arange(0, seq_len, device=device)[None]
            inv_freq_expanded = self.div_term[None, :, None].expand(position_ids.shape[0], -1, 1)
            position_ids_expanded = position_ids[:, None, :]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()[:, :, None, :]
            sin = emb.sin()[:, :, None, :]
            return cos, sin

    def forward(self, x, start_pos=0, weights=None, **kwargs):
        """x: (b s n d)
        y_{d-1} = x_{d-1}cos(w_{d/2}) - x_{d}sin(w_{d/2}), d in {1,3,5,...}
        y_{d} = x_{d}cos(w_{d/2}) + x_{d-1}sin(w_{d/2}), d in {2,4,6,...}
        """
        if weights is None:
            weights = self.make_weights(x.shape[1])

        cos, sin = weights
        # cos = cos.to(x)
        # sin = sin.to(x)
        s = x.shape[1]
        cos = cos[:, start_pos:start_pos + s, :, :]
        sin = sin[:, start_pos:start_pos + s, :, :]

        _x = x.type_as(cos)
        y = (_x * cos) + (self.rotate_half(_x) * sin)
        return y.type_as(x)

    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)


class QwenSdpaAttention(nn.Module):
    """cross attention"""

    def __init__(self, n_heads=None, model_dim=None, head_dim=None, n_kv_heads=None,
                 drop_prob=0.1, attend=None, out_layer=None, **fn_kwargs):
        super().__init__()
        n_heads, model_dim, head_dim = attentions.get_attention_input(n_heads, model_dim, head_dim)
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

        self.attend = QwenSdpaAttendWrapper(drop_prob=drop_prob) if attend is None else attend

    def forward(self, q, k=None, v=None, attention_mask=None, **attend_kwargs):
        q, k, v = attentions.get_qkv(q, k, v)
        q, k, v = [m(x) for m, x in zip(self.to_qkv, (q, k, v))]

        q = self.q_view_in(q).contiguous()
        k = self.kv_view_in(k).contiguous()
        v = self.kv_view_in(v).contiguous()

        x = self.attend(q, k, v, attention_mask=attention_mask, **attend_kwargs)
        x = self.view_out(x)

        x = self.to_out(x)
        return x


class QwenSdpaAttendWrapper(attentions.RotaryAttendWrapper):
    """vision scaled-dot-product-attention"""

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
        # note, mainly difference from `RotaryAttendWrapper`.
        k = k.repeat_interleave(ratio, dim=1)
        v = v.repeat_interleave(ratio, dim=1)
        attn = self.base_layer(q, k, v, attention_mask=attention_mask, **attend_kwargs)
        return attn
