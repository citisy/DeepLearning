import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_parse.nl_data_parse.pre_process.decoder import beam_search
from utils import torch_utils
from . import llama
from .transformers import PositionWiseFeedForward, TransformerSequential
from .. import activations, attentions, bundles, embeddings


class Config(bundles.Config):
    default_model = 'base'

    @classmethod
    def make_full_config(cls):
        # https://github.com/google-research/text-to-text-transfer-transformer?tab=readme-ov-file#released-model-checkpoints
        config_dict = {
            'small': dict(
                hidden_size=512,
                num_hidden_layers=6,
                num_attention_heads=8
            ),

            'base': dict(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12
            ),

            'large': dict(
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=16
            ),

            '3B': dict(
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=32
            ),

            '11B': dict(
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=128
            )
        }
        return config_dict


class WeightLoader(bundles.WeightLoader):
    @classmethod
    def auto_download(cls, save_path, save_name=''):
        # download weight auto from transformers
        from transformers import T5ForConditionalGeneration

        model = T5ForConditionalGeneration.from_pretrained(save_path)
        state_dict = model.state_dict()
        return state_dict


class WeightConverter:
    @staticmethod
    def from_hf(state_dict):
        """convert weights from huggingface model to my own model

        Usage:
            .. code-block:: python

                state_dict = torch.load('...')
                state_dict = convert_hf_weights(state_dict)
                Model(...).load_state_dict(state_dict)

                from transformers import BertForPreTraining
                state_dict = BertForPreTraining.from_pretrained('...')
                state_dict = convert_hf_weights(state_dict)
                Model(...).load_state_dict(state_dict)
        """
        convert_dict = {
            'shared': 'embedding',

            '{1}.block.{0}.layer.0.SelfAttention.q': '{1}.{0}.attn_res.fn.to_qkv.0',
            '{1}.block.{0}.layer.0.SelfAttention.k': '{1}.{0}.attn_res.fn.to_qkv.1',
            '{1}.block.{0}.layer.0.SelfAttention.v': '{1}.{0}.attn_res.fn.to_qkv.2',
            '{1}.block.{0}.layer.0.SelfAttention.o': '{1}.{0}.attn_res.fn.to_out.linear',
            '{1}.block.{0}.layer.0.layer_norm': '{1}.{0}.attn_res.norm',

            'encoder.block.{0}.layer.1.layer_norm': 'encoder.{0}.ff_res.norm',

            'decoder.block.{0}.layer.1.EncDecAttention.q': 'decoder.{0}.de_attn_res.fn.to_qkv.0',
            'decoder.block.{0}.layer.1.EncDecAttention.k': 'decoder.{0}.de_attn_res.fn.to_qkv.1',
            'decoder.block.{0}.layer.1.EncDecAttention.v': 'decoder.{0}.de_attn_res.fn.to_qkv.2',
            'decoder.block.{0}.layer.1.EncDecAttention.o': 'decoder.{0}.de_attn_res.fn.to_out.linear',
            'decoder.block.{0}.layer.1.layer_norm': 'decoder.{0}.de_attn_res.norm',

            'decoder.block.{0}.layer.2.layer_norm': 'decoder.{0}.ff_res.norm',

            '{0}.block.0.layer.0.SelfAttention.relative_attention_bias': '{0}_relative_bias',
            '{0}.final_layer_norm': '{0}_norm',

        }

        if 'encoder.block.0.layer.1.DenseReluDense.wi.weight' in state_dict:
            # without gated_act
            convert_dict.update({
                'encoder.block.{0}.layer.1.DenseReluDense.wi': 'encoder.{0}.ff_res.fn.0.linear',
                'encoder.block.{0}.layer.1.DenseReluDense.wo': 'encoder.{0}.ff_res.fn.1.linear',
                'decoder.block.{0}.layer.2.DenseReluDense.wi': 'decoder.{0}.ff_res.fn.0.linear',
                'decoder.block.{0}.layer.2.DenseReluDense.wo': 'decoder.{0}.ff_res.fn.1.linear',
            })
        else:
            # with gated_act
            convert_dict.update({
                'encoder.block.{0}.layer.1.DenseReluDense.wi_0': 'encoder.{0}.ff_res.fn.f1.linear',
                'encoder.block.{0}.layer.1.DenseReluDense.wi_1': 'encoder.{0}.ff_res.fn.f3.linear',
                'encoder.block.{0}.layer.1.DenseReluDense.wo': 'encoder.{0}.ff_res.fn.f2.linear',
                'decoder.block.{0}.layer.2.DenseReluDense.wi_0': 'decoder.{0}.ff_res.fn.f1.linear',
                'decoder.block.{0}.layer.2.DenseReluDense.wi_1': 'decoder.{0}.ff_res.fn.f3.linear',
                'decoder.block.{0}.layer.2.DenseReluDense.wo': 'decoder.{0}.ff_res.fn.f2.linear',
            })

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        return state_dict


class Model(nn.Module):
    """refer to
    paper:
        - Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    code:
        - https://github.com/google-research/text-to-text-transfer-transformer
    this code base on version of `transformers`
    """

    def __init__(
            self, vocab_size=32128, eos_id=1,
            hidden_size=768, num_attention_heads=12,
            ff_hidden_size=None, is_gated_act=False, ff_act_type='ReLU',
            num_hidden_layers=12,
            n_relative_buckets=32, relative_max_distance=128,
            drop_prob=0.1):
        super().__init__()
        ff_hidden_size = ff_hidden_size or hidden_size * 4

        self.hidden_size = hidden_size
        self.eos_id = eos_id

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        fn_kwargs = dict(
            qkv_fn_kwargs=dict(
                bias=False
            ),
            out_fn_kwargs=dict(
                bias=False
            )
        )
        attend_kwargs = dict(
            n_relative_buckets=n_relative_buckets,
            relative_max_distance=relative_max_distance,
            n_heads=num_attention_heads
        )
        ff_kwargs = dict(
            act=activations.make_act_fn.get(ff_act_type)(),
            bias=False,
        )
        self.encoder_relative_bias = nn.Embedding(n_relative_buckets, num_attention_heads)
        self.encoder = TransformerSequential(
            hidden_size, num_attention_heads, ff_hidden_size,
            norm_first=True,
            norm_fn=T5LayerNorm,
            drop_prob=drop_prob,
            fn_kwargs=fn_kwargs,
            attend=T5ScaleAttend(
                relative_bias=self.encoder_relative_bias,
                is_relative_bidirectional=True,
                **attend_kwargs
            ),
            feed_forward_fn=llama.FeedForward if is_gated_act else PositionWiseFeedForward,
            ff_kwargs=ff_kwargs,
            num_blocks=num_hidden_layers
        )
        self.encoder_norm = T5LayerNorm(hidden_size)

        self.decoder_relative_bias = nn.Embedding(n_relative_buckets, num_attention_heads)
        self.decoder = TransformerSequential(
            hidden_size, num_attention_heads, hidden_size * 4,
            is_decode=True,
            norm_first=True,
            norm_fn=T5LayerNorm,
            drop_prob=drop_prob,
            fn_kwargs=fn_kwargs,
            de_fn_kwargs=fn_kwargs,
            attend=T5ScaleAttend(
                relative_bias=self.decoder_relative_bias,
                is_relative_bidirectional=False,
                **attend_kwargs
            ),
            de_attend=T5ScaleAttend(),
            feed_forward_fn=llama.FeedForward if is_gated_act else PositionWiseFeedForward,
            ff_kwargs=ff_kwargs,
            num_blocks=num_hidden_layers
        )
        self.decoder_norm = T5LayerNorm(hidden_size)

        self.head = embeddings.EmbeddingSim(self.embedding.weight)

    def set_encoder_only(self):
        del self.decoder_relative_bias
        del self.decoder
        del self.decoder_norm
        del self.head

    def forward(self, *args, **kwargs):
        if self.training:
            return self.fit(*args, **kwargs)
        else:
            return self.inference(*args, **kwargs)

    def fit(self, x, attention_mask=None, **kwargs):
        context = self.encode(x, attention_mask=attention_mask)
        # note, shift one token to predict the future word
        trues = torch.cat([x[:, 1:], torch.full((len(x), 1), self.pad_id)], dim=1)
        logits = self.decode(x, context=context, context_mask=attention_mask, **kwargs)
        loss = self.loss(logits, trues)
        return {'loss': loss}

    def loss(self, logits, trues):
        logits = logits.transpose(1, 2)  # seq first -> class first
        return F.cross_entropy(logits, trues)

    def inference(self, text_ids, y=None, seq_lens=None, attention_mask=None, **kwargs):
        context = self.encode(text_ids, attention_mask=attention_mask)
        if y is None:
            y = torch.zeros((len(text_ids), 1), dtype=torch.long, device=text_ids.device)
            seq_lens = [1] * len(text_ids)
        preds = self.post_process(y, context=context, context_mask=attention_mask, seq_lens=seq_lens, **kwargs)
        return {'preds': preds}

    def encode(self, x, attention_mask=None, **encoder_kwargs):
        x = self.embedding(x)
        x = self.encoder(x, attention_mask=attention_mask, **encoder_kwargs)
        x = self.encoder_norm(x)
        return x

    def decode(self, x, **decoder_kwargs):
        attention_mask = attentions.make_causal_attention_mask(x)
        x = self.embedding(x)
        x = self.decoder(x, attention_mask=attention_mask, **decoder_kwargs)
        x = self.decoder_norm(x)
        # note, don't scale in attention but here
        x = x * (self.hidden_size ** -0.5)
        x = self.head(x)
        return x

    def post_process(self, x, **decode_kwargs):
        return self.decode(x, **decode_kwargs)


class T5LayerNorm(nn.Module):
    """copy from `transformers`"""

    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class T5ScaleAttend(attentions.ScaleAttend):
    """ScaleAttend from T5"""

    def __init__(
            self, drop_prob=0.,
            relative_bias=None, n_relative_buckets=None, n_heads=None,
            relative_max_distance=None, is_relative_bidirectional=False,
    ):
        super().__init__(drop_prob=drop_prob)
        self.relative_bias = relative_bias
        self.n_relative_buckets = n_relative_buckets
        self.n_heads = n_heads
        self.relative_max_distance = relative_max_distance
        self.is_relative_bidirectional = is_relative_bidirectional

    def forward(self, q, k, v, attention_mask=None, **kwargs):
        # similarity -> (..., i, j), usually i=j=s
        # sim = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scale
        # note, don't use scaling from t5 attention
        sim = torch.matmul(q, k.transpose(-2, -1))

        if self.relative_bias is not None:
            assert sim.ndim == 4
            bias = self.make_relative_bias(sim)
            sim += bias

        sim = attentions.mask_values(sim, attention_mask)
        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)

        # attn = einsum('... i j, ... j d -> ... i d', attn, v)
        attn = torch.matmul(attn, v)

        return attn

    def make_relative_bias(self, sim):
        b, n, i, j = sim.shape
        device = sim.device

        context_position = torch.arange(i, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(j, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # (i, j)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.is_relative_bidirectional,
            num_buckets=self.n_relative_buckets,
            max_distance=self.relative_max_distance,
        )
        values = self.relative_bias(relative_position_bucket)  # (i, j, n)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # (b, n, i, j)
        return values

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """copy from transformers
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
