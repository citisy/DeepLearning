import math
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from utils import torch_utils
from .. import attentions, embeddings
from ..layers import Linear
from ..text_pretrain import transformers


class WeightConverter:
    @classmethod
    def from_official(cls, state_dict):
        convert_dict = {
            'encoder.{0}.feed_forward.w_1': 'encoder.{0}.feed_forward.0.linear',
            'encoder.{0}.feed_forward.w_2': 'encoder.{0}.feed_forward.1.linear',
            'encoder.{0}.feed_forward.norm': 'encoder.{0}.feed_forward.0.norm',
            'encoder.{0}.norm1': 'encoder.{0}.norm1',
            'encoder.{0}.norm2': 'encoder.{0}.norm2',
            'encoder.{0}.self_attn.linear_q_k_v': 'encoder.{0}.self_attn.to_qkv',
            'encoder.{0}.self_attn.linear_out': 'encoder.{0}.self_attn.to_out.linear',
            'encoder.{0}.self_attn.fsmn_block': 'encoder.{0}.self_attn.fsmn_block',

            'decoder.{0}.feed_forward.w_1': 'decoder.{0}.feed_forward.0.linear',
            'decoder.{0}.feed_forward.w_2': 'decoder.{0}.feed_forward.1.linear',
            'decoder.{0}.feed_forward.norm': 'decoder.{0}.feed_forward.0.norm',

            'predictor': 'head'
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, eps=1e-12, **kwargs)


class Model(nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    code: `funasr.models.bicif_paraformer.model.Paraformer`
    """

    input_size: int = 560
    vocab_size: int = 8404
    ignore_id: int = -1
    blank_id: int = 0
    sos: int = 1
    eos: int = 2

    def __init__(
            self,
            encoder_config={}, decoder_config={}, head_config={}, model_config={},
            **kwargs,
    ):
        super().__init__()
        self.__dict__.update(model_config)
        self.encoder = SANMEncoder(input_size=self.input_size, **encoder_config)
        self.head = CifPredictorV3(**head_config)
        self.decoder = SANMDecoder(
            vocab_size=self.vocab_size,
            encoder_output_size=self.encoder.output_size,
            **decoder_config,
        )

    def forward(self, x, **kwargs):
        if self.training:
            raise NotImplementedError
        else:
            return self.post_process(x, **kwargs)

    def post_process(self, *args, **kwargs):
        raise NotImplementedError

    def encode(
            self,
            x: torch.Tensor,
            seq_lens: torch.Tensor,
            **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (Batch, Length, ...)
            seq_lens: (Batch, )
        """
        # Forward encoder
        encoder_out, encoder_out_lens, _ = self.encoder(x, seq_lens)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens


class SANMEncoder(nn.Module):
    """
    Author: Zhifu Gao, Shiliang Zhang, Ming Lei, Ian McLoughlin
    San-m: Memory equipped self-attention for end-to-end speech recognition
    https://arxiv.org/abs/2006.01713
    code: `funasr.models.sanm.encoder.SANMEncoder`
    """

    def __init__(
            self,
            input_size: int = 560,
            output_size: int = 512,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 50,
            dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.1,
            norm_first: bool = True,
            concat_after: bool = False,
            kernel_size: int = 11,
            sanm_shfit: int = 0,
            **kwargs
    ):
        super().__init__()
        self.output_size = output_size

        self.embed = SinusoidalEmbedding(input_size)

        self.encoders0 = nn.ModuleList([SANMEncoderLayer(
            input_size,
            output_size,
            SANMEncoderAttention(
                n_heads=attention_heads,
                query_dim=input_size,
                context_dim=input_size,
                model_dim=output_size,
                separate=False,
                drop_prob=attention_dropout_rate,
                kernel_size=kernel_size,
                sanm_shfit=sanm_shfit,
                out_layer=Linear(output_size, output_size, mode='ld', drop_prob=attention_dropout_rate)
            ),
            transformers.PositionWiseFeedForward(
                output_size,
                linear_units,
                nn.ReLU(),
                dropout_rate,
            ),
            dropout_rate,
            norm_first,
            concat_after,
        ) for _ in range(1)])

        self.encoders = nn.ModuleList([
            SANMEncoderLayer(
                output_size,
                output_size,
                SANMEncoderAttention(
                    n_heads=attention_heads,
                    query_dim=output_size,
                    context_dim=output_size,
                    model_dim=output_size,
                    separate=False,
                    drop_prob=attention_dropout_rate,
                    kernel_size=kernel_size,
                    sanm_shfit=sanm_shfit,
                    out_layer=Linear(output_size, output_size, mode='ld', drop_prob=attention_dropout_rate)
                ),
                transformers.PositionWiseFeedForward(
                    output_size,
                    linear_units,
                    nn.ReLU(),
                    dropout_rate,
                ),
                dropout_rate,
                norm_first,
                concat_after,
            ) for _ in range(num_blocks - 1)
        ])
        self.after_norm = LayerNorm(output_size)

        self.conditioning_layer = None
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
            self,
            xs_pad: torch.Tensor,
            ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
        Returns:
            position embedded tensor and mask
        """
        masks = (attentions.make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        xs_pad = xs_pad * self.output_size ** 0.5
        xs_pad += self.embed(torch.arange(1, xs_pad.shape[1] + 1, device=xs_pad.device, dtype=torch.float32))[None]

        # xs_pad = self.dropout(xs_pad)
        args = (xs_pad, masks)
        for m in self.encoders0:
            args = m(*args)
        xs_pad, masks = args[0], args[1]
        args = (xs_pad, masks)
        for m in self.encoders:
            args = m(*args)
        xs_pad, masks = args[0], args[1]
        xs_pad = self.after_norm(xs_pad)
        olens = masks.squeeze(1).sum(1)

        return xs_pad, olens, None


class SinusoidalEmbedding(embeddings.SinusoidalEmbedding):
    def _register(self):
        div_term = (torch.arange(self.embedding_dim / 2).float() * -(math.log(self.theta) / (self.embedding_dim / 2 - 1))).exp()
        self.register_buffer('div_term', div_term, persistent=False)

    def forward(self, x):
        """

        Args:
            x: (n_position, )

        """
        dtype = x.dtype
        div_term = self.div_term
        x = x * self.factor
        emb = x[:, None].float() * div_term[None, :]
        # note, different here
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = emb.to(dtype)
        return emb


class SANMEncoderLayer(nn.Module):
    def __init__(
            self,
            in_size,
            size,
            self_attn,
            feed_forward,
            drop_prob,
            norm_first=True,
            is_concat=False,
            stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(in_size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(drop_prob)
        self.in_size = in_size
        self.size = size
        self.norm_first = norm_first
        # if True, x -> x + linear(concat(x, att(x)))
        # if False, x -> x + att(x)
        self.is_concat = is_concat
        if self.is_concat:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate
        self.dropout_rate = drop_prob

    def forward(self, x, mask, cache=None, mask_shfit_chunk=None, mask_att_chunk_encoder=None):
        """Compute encoded features.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        stoch_layer_coeff = 1.0

        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

            if skip_layer:
                if cache is not None:
                    x = torch.cat([cache, x], dim=1)
                return x, mask

        residual = x
        if self.norm_first:
            x = self.norm1(x)

        if self.is_concat:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x,
                        attention_mask=mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    ),
                ),
                dim=-1,
            )
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            if self.in_size == self.size:
                x = residual + stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        attention_mask=mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        attention_mask=mask,
                        mask_shfit_chunk=mask_shfit_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
        if not self.norm_first:
            x = self.norm1(x)

        residual = x
        if self.norm_first:
            x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))
        if not self.norm_first:
            x = self.norm2(x)

        return x, mask, cache, mask_shfit_chunk, mask_att_chunk_encoder


class FSMNBlock(nn.Module):
    def __init__(self, *args, kernel_size=11, sanm_shfit=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fsmn_block = nn.Conv1d(
            self.model_dim, self.model_dim, kernel_size, stride=1, padding=0, groups=self.model_dim, bias=False
        )

        # padding
        left_padding = (kernel_size - 1) // 2
        if sanm_shfit > 0:
            left_padding = left_padding + sanm_shfit
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shfit_chunk is not None:
                mask = mask * mask_shfit_chunk
            inputs = inputs * mask

        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        # x = self.dropout(x)
        if mask is not None:
            x = x * mask
        return x


class SANMEncoderAttention(FSMNBlock, attentions.CrossAttention2D):

    def forward(self, q, k=None, v=None, attention_mask=None, mask_shfit_chunk=None, **attend_kwargs):
        dim = 1 if self.use_conv else -1
        qkv = self.to_qkv(q)
        q, k, v = qkv.chunk(3, dim=dim)

        fsmn_memory = self.forward_fsmn(v, attention_mask, mask_shfit_chunk)

        q, k, v = [self.view_in(x).contiguous() for x in (q, k, v)]

        attention_mask = attention_mask[:, None]  # (b, 1, 1, n)
        x = self.attend(q, k, v, attention_mask=attention_mask, **attend_kwargs)
        x = self.forward_out(x)
        x = x + fsmn_memory
        return x


class SANMDecoder(nn.Module):
    """
    code: `funasr.models.e_paraformer.decoder.ParaformerSANMDecoder`
    """

    def __init__(
            self,
            vocab_size,
            encoder_output_size,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 16,
            dropout_rate: float = 0.1,
            src_attention_dropout_rate: float = 0.1,
            use_output_layer: bool = True,
            att_layer_num: int = 16,
            kernel_size: int = 11,
            sanm_shfit: int = 0,
            **kwargs
    ):
        super().__init__()

        attention_dim = encoder_output_size
        self.embed = nn.Sequential(
            nn.Embedding(vocab_size, attention_dim),
        )

        self.after_norm = LayerNorm(attention_dim)
        self.output_layer = nn.Linear(attention_dim, vocab_size) if use_output_layer else nn.Identity()

        self.att_layer_num = att_layer_num
        self.num_blocks = num_blocks
        if sanm_shfit is None:
            sanm_shfit = (kernel_size - 1) // 2
        self.decoders = nn.ModuleList([SANMDecoderLayer(
            attention_dim,
            SANMDecoderAttention(
                model_dim=attention_dim,
                kernel_size=kernel_size,
                sanm_shfit=sanm_shfit
            ),
            CrossAttention2D(
                attention_heads,
                attention_dim,
                src_attention_dropout_rate,
            ),
            transformers.PositionWiseFeedForward(
                attention_dim, linear_units, nn.ReLU(), dropout_rate,
                l1_kwargs=dict(
                    mode='lan', norm_fn=LayerNorm
                ),
                l2_kwargs=dict(
                    linear_fn=partial(nn.Linear, bias=False)
                )
            ),
            dropout_rate,
        ) for _ in range(att_layer_num)])

        self.decoders3 = nn.ModuleList([SANMDecoderLayer(
            attention_dim,
            None,
            None,
            transformers.PositionWiseFeedForward(
                attention_dim, linear_units, nn.ReLU(), dropout_rate,
                l1_kwargs=dict(
                    mode='lan', norm_fn=LayerNorm
                ),
                l2_kwargs=dict(
                    linear_fn=partial(nn.Linear, bias=False)
                )
            ),
            dropout_rate,
        ) for _ in range(1)])

    def forward(
            self,
            hs_pad: torch.Tensor,
            hlens: torch.Tensor,
            ys_in_pad: torch.Tensor,
            ys_in_lens: torch.Tensor,
            chunk_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        tgt = ys_in_pad
        tgt_mask = attentions.make_pad_mask(ys_in_lens).to(device=tgt.device)

        memory = hs_pad
        memory_mask = attentions.make_pad_mask(hlens).to(device=memory.device)
        if chunk_mask is not None:
            memory_mask = memory_mask * chunk_mask
            if tgt_mask.size(1) != memory_mask.size(1):
                memory_mask = torch.cat((memory_mask, memory_mask[:, -2:-1, :]), dim=1)

        x = tgt
        for m in self.decoders:
            x, tgt_mask, memory, memory_mask, _ = m(x, tgt_mask, memory, memory_mask)
        for m in self.decoders3:
            x, tgt_mask, memory, memory_mask, _ = m(x, tgt_mask, memory, memory_mask)
        hidden = self.after_norm(x)
        x = self.output_layer(hidden)
        return x


class SANMDecoderLayer(nn.Module):
    """Single decoder layer module."""

    def __init__(
            self,
            size,
            self_attn,
            src_attn,
            feed_forward,
            drop_prob,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        if self_attn is not None:
            self.norm2 = LayerNorm(size)
        if src_attn is not None:
            self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, tgt, tgt_mask, context, context_mask=None, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            context (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, size).
            context_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).
            cache (List[torch.Tensor]): List of cached tensors.
                Each tensor shape should be (#batch, maxlen_out - 1, size).

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn is not None:
            tgt = self.norm2(tgt)
            x = self.self_attn(tgt, tgt_mask)
            x = residual + self.dropout(x)

        if self.src_attn is not None:
            residual = x
            x = self.norm3(x)
            x_src_attn = self.src_attn(x, context, context_mask)
            x = residual + self.dropout(x_src_attn)

        return x, tgt_mask, context, context_mask, cache


class SANMDecoderAttention(FSMNBlock):
    """actually it has no attention layers"""

    def __init__(self, model_dim, **kwargs):
        self.model_dim = model_dim
        super().__init__(**kwargs)

    def forward(self, x, attention_mask=None, mask_shfit_chunk=None):
        return self.forward_fsmn(x, attention_mask, mask_shfit_chunk)


class CrossAttention2D(nn.Module):
    def __init__(
            self,
            n_heads,
            model_dim,
            drop_prob,
            context_dim=None,
    ):
        super().__init__()
        assert model_dim % n_heads == 0
        self.head_dim = model_dim // n_heads
        self.n_heads = n_heads
        # note, only one difference from attentions.CrossAttention2D
        # here merging kv linear not qkv linear
        # and I don't want to compatible with this code yet
        self.linear_q = nn.Linear(model_dim, model_dim)
        self.linear_k_v = nn.Linear(
            model_dim if context_dim is None else context_dim, model_dim * 2
        )

        self.view_in = Rearrange('b s (n dk)-> b n s dk', n=n_heads)
        self.attend = attentions.ScaleAttend(drop_prob=drop_prob)
        self.view_out = Rearrange('b n s dk -> b s (n dk)')
        self.linear_out = nn.Linear(model_dim, model_dim)

    def forward_in(self, x, context):
        q = self.linear_q(x)
        k, v = self.linear_k_v(context).chunk(2, dim=-1)
        q, k, v = [self.view_in(x).contiguous() for x in (q, k, v)]
        return q, k, v

    def forward_out(self, x, *args, **kwargs):
        x = self.view_out(x)
        x = self.linear_out(x)
        return x

    def forward(self, q, context, attention_mask=None, **attend_kwargs):
        q, k, v = self.forward_in(q, context)
        x = self.attend(q, k, v, attention_mask=attention_mask, **attend_kwargs)
        x = self.forward_out(x)
        return x


class CifPredictorV3(nn.Module):
    """code: `funasr.models.bicif_paraformer.cif_predictor.CifPredictorV3`"""
    input_dim = 512
    l_order = 1
    r_order = 1
    threshold = 1.0
    drop_prob = 0.1
    smooth_factor = 1.0
    noise_threshold = 0
    tail_threshold = 0.45
    smooth_factor2 = 0.25
    noise_threshold2 = 0.01
    upsample_times = 3
    use_cif1_cnn = False

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        input_dim = self.input_dim
        l_order = self.l_order
        r_order = self.r_order

        self.pad = nn.ConstantPad1d((l_order, r_order), 0)
        self.cif_conv1d = nn.Conv1d(input_dim, input_dim, l_order + r_order + 1)
        self.cif_output = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(p=self.drop_prob)
        self.upsample_cnn = nn.ConvTranspose1d(input_dim, input_dim, self.upsample_times, self.upsample_times)
        self.blstm = nn.LSTM(input_dim, input_dim, 1, bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        self.cif_output2 = nn.Linear(input_dim * 2, 1)

    def forward(
            self,
            hidden,
            target_label=None,
            mask=None,
            ignore_id=-1,
            mask_chunk_predictor=None,
            target_label_length=None,
    ):
        h = hidden
        mk = mask

        context = hidden.transpose(1, 2)
        queries = self.pad(context)
        output = F.relu(self.cif_conv1d(queries))
        output = output.transpose(1, 2)

        output = self.cif_output(output)
        alphas = F.sigmoid(output)
        alphas = F.relu(alphas * self.smooth_factor - self.noise_threshold)
        if mk is not None:
            mk = mk.transpose(-1, -2).float()
            alphas = alphas * mk
        if mask_chunk_predictor is not None:
            alphas = alphas * mask_chunk_predictor
        alphas = alphas.squeeze(-1)
        mk = mk.squeeze(-1)
        if target_label_length is not None:
            target_length = target_label_length
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)

        if target_length is not None:
            alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            h, alphas, token_num = self.tail_process_fn(h, alphas, mask=mk)

        acoustic_embeds, _ = self.cif(h, alphas)
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]

        us_alphas, us_peaks = self.get_timestamp(hidden, mask, token_num)
        return acoustic_embeds, token_num, alphas, us_alphas, us_peaks

    def tail_process_fn(self, hidden, alphas, mask=None):
        b, t, d = hidden.size()
        tail_threshold = self.tail_threshold
        if mask is not None:
            zeros_t = torch.zeros((b, 1), dtype=torch.float32, device=alphas.device)
            ones_t = torch.ones_like(zeros_t)
            mask_1 = torch.cat([mask, zeros_t], dim=1)
            mask_2 = torch.cat([ones_t, mask], dim=1)
            mask = mask_2 - mask_1
            tail_threshold = mask * tail_threshold
            alphas = torch.cat([alphas, zeros_t], dim=1)
            alphas = torch.add(alphas, tail_threshold)
        else:
            tail_threshold = torch.tensor([tail_threshold], dtype=alphas.dtype).to(alphas.device)
            tail_threshold = torch.reshape(tail_threshold, (1, 1))
            alphas = torch.cat([alphas, tail_threshold], dim=1)
        zeros = torch.zeros((b, 1, d), dtype=hidden.dtype).to(hidden.device)
        hidden = torch.cat([hidden, zeros], dim=1)
        token_num = alphas.sum(dim=-1)
        token_num_floor = torch.floor(token_num)

        return hidden, alphas, token_num_floor

    def cif(self, hidden, alphas):
        batch_size, len_time, hidden_size = hidden.size()

        # loop varss
        integrate = torch.zeros([batch_size], device=hidden.device)
        frame = torch.zeros([batch_size, hidden_size], device=hidden.device)
        # intermediate vars along time
        list_fires = []
        list_frames = []

        for t in range(len_time):
            alpha = alphas[:, t]
            distribution_completion = torch.ones([batch_size], device=hidden.device) - integrate

            integrate += alpha
            list_fires.append(integrate)

            fire_place = integrate >= self.threshold
            integrate = torch.where(
                fire_place,
                integrate - torch.ones([batch_size], device=hidden.device),
                integrate
            )
            cur = torch.where(fire_place, distribution_completion, alpha)
            remainds = alpha - cur

            frame += cur[:, None] * hidden[:, t, :]
            list_frames.append(frame)
            frame = torch.where(
                fire_place[:, None].repeat(1, hidden_size), remainds[:, None] * hidden[:, t, :], frame
            )

        fires = torch.stack(list_fires, 1)
        frames = torch.stack(list_frames, 1)
        list_ls = []
        len_labels = torch.round(alphas.sum(-1)).int()
        max_label_len = len_labels.max()
        for b in range(batch_size):
            fire = fires[b, :]
            l = torch.index_select(frames[b, :, :], 0, torch.nonzero(fire >= self.threshold).squeeze())
            pad_l = torch.zeros([max_label_len - l.size(0), hidden_size], device=hidden.device)
            list_ls.append(torch.cat([l, pad_l], 0))
        return torch.stack(list_ls, 0), fires

    def get_timestamp(self, hidden, mask=None, token_num=None):
        context = hidden.transpose(1, 2)

        # an extra head for timestamp prediction
        if self.use_cif1_cnn:
            queries = self.pad(context)
            output = F.relu(self.cif_conv1d(queries))
        else:
            output = context

        output = self.upsample_cnn(output).transpose(1, 2)
        output, (_, _) = self.blstm(output)

        us_alphas = F.sigmoid(self.cif_output2(output))
        us_alphas = F.relu(us_alphas * self.smooth_factor2 - self.noise_threshold2)
        # repeat the mask in T demension to match the upsampled length
        if mask is not None:
            mask = (
                mask.repeat(1, self.upsample_times, 1)
                .transpose(-1, -2)
                .reshape(us_alphas.shape[0], -1)
                .unsqueeze(-1)
            )
            us_alphas = us_alphas * mask
        us_alphas = us_alphas.squeeze(-1)
        if token_num is not None:
            _token_num = us_alphas.sum(-1)
            us_alphas *= (token_num / _token_num)[:, None].repeat(1, us_alphas.size(1))

        # upsampled alphas and cif_peak
        us_cif_peak = self.cif_wo_hidden(us_alphas)
        return us_alphas, us_cif_peak

    def cif_wo_hidden(self, alphas):
        threshold = self.threshold - 1e-4
        batch_size, len_time = alphas.size()

        # loop varss
        integrate = torch.zeros([batch_size], device=alphas.device)
        # intermediate vars along time
        list_fires = []

        for t in range(len_time):
            alpha = alphas[:, t]

            integrate += alpha
            list_fires.append(integrate)

            fire_place = integrate >= threshold
            integrate = torch.where(
                fire_place,
                integrate - torch.ones([batch_size], device=alphas.device) * threshold,
                integrate,
            )

        fires = torch.stack(list_fires, 1)
        return fires
