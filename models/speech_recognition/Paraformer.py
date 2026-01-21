import math
from functools import partial

import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch import nn

from utils import torch_utils
from .. import attentions, embeddings
from ..layers import Linear
from ..losses import LabelSmoothingLoss, MAELoss
from ..text_pretrain import transformers
from ..bundles import WeightLoader


class WeightConverter:
    encoder_convert_dict = {
        'encoder.{0}.feed_forward.w_1': 'encoder.{0}.feed_forward.0.linear',
        'encoder.{0}.feed_forward.w_2': 'encoder.{0}.feed_forward.1.linear',
        'encoder.{0}.feed_forward.norm': 'encoder.{0}.feed_forward.0.norm',
        'encoder.{0}.norm1': 'encoder.{0}.norm1',
        'encoder.{0}.norm2': 'encoder.{0}.norm2',
        'encoder.{0}.self_attn.linear_q_k_v': 'encoder.{0}.self_attn.to_qkv',
        'encoder.{0}.self_attn.linear_out': 'encoder.{0}.self_attn.to_out.linear',
        'encoder.{0}.self_attn.fsmn_block': 'encoder.{0}.self_attn.fsmn_block',
    }

    decoder_convert_dict = {
        'decoder.{0}.feed_forward.w_1': 'decoder.{0}.feed_forward.0.linear',
        'decoder.{0}.feed_forward.w_2': 'decoder.{0}.feed_forward.1.linear',
        'decoder.{0}.feed_forward.norm': 'decoder.{0}.feed_forward.0.norm',
    }

    @classmethod
    def from_official(cls, state_dict):
        convert_dict = {
            **cls.encoder_convert_dict,
            **cls.decoder_convert_dict,
            'predictor': 'neck',
            'ctc.': 'criterion_ctc.'
        }
        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, eps=1e-12, **kwargs)


class Model(nn.Module):
    """
    refer to:
    paper:
        - (Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition)[https://arxiv.org/abs/2206.08317]
    code:
        - `funasr.models.bicif_paraformer.model.Paraformer`
    blog:
        - https://mp.weixin.qq.com/s/xQ87isj5_wxWiQs4qUXtVw
    """

    vocab_size: int = 8404
    ignore_id: int = -1
    blank_id: int = 0
    sos_id: int = 1
    eos_id: int = 2

    predictor_weight: float = 1.0
    use_ctc = False
    ctc_weight: float = 0.
    lsm_weight: float = 0.0
    length_normalized_loss: bool = False
    sampling_ratio = 0.75
    share_embedding: bool = False

    def __init__(
            self,
            encoder_config={}, decoder_config={}, head_config={}, model_config={},
            ctc_config={},
            **kwargs,
    ):
        super().__init__()
        self.__dict__.update(model_config)
        self.make_encoder(**encoder_config)
        self.make_neck(**head_config)
        self.make_decoder(**decoder_config)

        if self.use_ctc:
            self.criterion_ctc = CTC(odim=self.vocab_size, encoder_output_size=self.encoder.output_size, **ctc_config)
        else:
            self.criterion_ctc = None
        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.ignore_id,
            smoothing=self.lsm_weight,
            normalize_length=self.length_normalized_loss,
        )
        self.criterion_pre = MAELoss(normalize_length=self.length_normalized_loss)

    def make_encoder(self, **encoder_config):
        self.encoder = SANMEncoder(**encoder_config)

    def make_neck(self, **head_config):
        self.neck = CifPredictorV2(**head_config)

    def make_decoder(self, **decoder_config):
        self.decoder = SANMDecoder(
            vocab_size=self.vocab_size,
            encoder_output_size=self.encoder.output_size,
            **decoder_config,
        )

    def _calc_ctc_loss(
            self,
            encoder_out: torch.Tensor,
            encoder_out_lens: torch.Tensor,
            ys_pad: torch.Tensor,
            ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.criterion_ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        return loss_ctc

    def forward(self, *args, **kwargs):
        if self.training:
            return self.fit(*args, **kwargs)
        else:
            return self.inference(*args, **kwargs)

    def fit(self, speech, speech_lens, **kwargs):
        encoder_out, encoder_out_mask = self.process(speech, speech_lens)
        return self.loss(encoder_out, encoder_out_mask, speech_lens, **kwargs)

    def inference(self, speech, speech_lens, **kwargs):
        encoder_out, encoder_out_mask = self.process(speech, speech_lens)
        return self.post_process(encoder_out, encoder_out_mask, speech_lens, **kwargs)

    def process(self, speech, speech_lens):
        encoder_out, encoder_out_mask = self.encode(speech, speech_lens)
        return encoder_out, encoder_out_mask

    def encode(self, speech: torch.Tensor, speech_lens: torch.Tensor, **kwargs):
        attention_mask = attentions.make_pad_mask(speech_lens)[:, None, :].to(speech.device)
        return self.encoder(speech, speech_lens, attention_mask), attention_mask

    def post_process(self, encoder_out, encoder_out_mask, speech_lens, **kwargs):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        raise NotImplementedError


class SANMEncoder(nn.Module):
    """
    refer to:
    paper:
        (San-m: Memory equipped self-attention for end-to-end speech recognition)[https://arxiv.org/abs/2006.01713]
    code:
        `funasr.models.sanm.encoder.SANMEncoder`
    """
    input_size: int = 560
    output_size: int = 512
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 50
    drop_prob: float = 0.1
    attention_drop_prob: float = 0.1
    norm_first: bool = True
    concat_after: bool = False
    kernel_size: int = 11
    sanm_shfit: int = 0

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        super().__init__()
        self.embed = SinusoidalEmbedding(self.input_size)

        self.encoders0 = nn.ModuleList([self.make_sanm_encoder(self.input_size, self.output_size) for _ in range(1)])

        self.encoders = nn.ModuleList([self.make_sanm_encoder(self.output_size, self.output_size) for _ in range(self.num_blocks - 1)])
        self.after_norm = LayerNorm(self.output_size)

        self.conditioning_layer = None
        self.dropout = nn.Dropout(self.drop_prob)

    def make_sanm_encoder(self, input_size, output_size):
        return SANMEncoderLayer(
            input_size,
            output_size,
            SANMEncoderAttention(
                n_heads=self.attention_heads,
                query_dim=input_size,
                context_dim=input_size,
                model_dim=output_size,
                separate=False,
                drop_prob=self.attention_drop_prob,
                kernel_size=self.kernel_size,
                sanm_shfit=self.sanm_shfit,
                out_layer=Linear(output_size, output_size, mode='ld', drop_prob=self.attention_drop_prob)
            ),
            transformers.PositionWiseFeedForward(
                output_size,
                self.linear_units,
                nn.ReLU(),
                self.drop_prob,
            ),
            self.drop_prob,
            self.norm_first,
            self.concat_after,
        )

    def forward(self, x, seq_lens, attention_mask):
        """Embed positions in tensor.

        Args:
            x: input tensor (B, L, D)
            seq_lens: input length (B)
            attention_mask:
        Returns:
            position embedded tensor and mask
        """
        x = x * self.output_size ** 0.5
        x += self.embed(torch.arange(1, x.shape[1] + 1, device=x.device, dtype=torch.float32))[None]

        x = self.dropout(x)
        for m in self.encoders0:
            x = m(x, attention_mask)

        for m in self.encoders:
            x = m(x, attention_mask)

        x = self.after_norm(x)
        return x


class SinusoidalEmbedding(embeddings.SinusoidalEmbedding):
    def initialize_layers(self):
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

    def forward(self, x, attention_mask, cache=None, mask_shift_chunk=None, mask_att_chunk_encoder=None):
        """Compute encoded features.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, size).
            attention_mask (torch.Tensor): Mask tensor for the input (#batch, time).
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
                return x

        residual = x
        if self.norm_first:
            x = self.norm1(x)

        if self.is_concat:
            x_concat = torch.cat(
                (
                    x,
                    self.self_attn(
                        x,
                        attention_mask=attention_mask,
                        mask_shift_chunk=mask_shift_chunk,
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
                        attention_mask=attention_mask,
                        mask_shift_chunk=mask_shift_chunk,
                        mask_att_chunk_encoder=mask_att_chunk_encoder,
                    )
                )
            else:
                x = stoch_layer_coeff * self.dropout(
                    self.self_attn(
                        x,
                        attention_mask=attention_mask,
                        mask_shift_chunk=mask_shift_chunk,
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

        return x


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

    def forward_fsmn(self, inputs, mask, mask_shift_chunk=None):
        b, t, d = inputs.size()
        if mask is not None:
            mask = torch.reshape(mask, (b, -1, 1))
            if mask_shift_chunk is not None:
                mask = mask * mask_shift_chunk
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

    def forward(self, q, k=None, v=None, attention_mask=None, mask_shift_chunk=None, **attend_kwargs):
        dim = 1 if self.use_conv else -1
        qkv = self.to_qkv(q)
        q, k, v = qkv.chunk(3, dim=dim)

        fsmn_memory = self.forward_fsmn(v, attention_mask, mask_shift_chunk)

        q, k, v = [self.view_in(x).contiguous() for x in (q, k, v)]

        attention_mask = attention_mask[:, None]  # (b, 1, 1, n)
        x = self.attend(q, k, v, attention_mask=attention_mask, **attend_kwargs)
        x = self.forward_out(x)
        x = x + fsmn_memory
        return x


class SANMDecoder(nn.Module):
    """
    code: `funasr.models.paraformer.decoder.ParaformerSANMDecoder`
    """

    def __init__(
            self,
            vocab_size,
            encoder_output_size,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 16,
            drop_prob: float = 0.1,
            src_attention_dropout_rate: float = 0.1,
            use_embed: bool = True,
            use_output_layer: bool = True,
            att_layer_num: int = 16,
            kernel_size: int = 11,
            sanm_shfit: int = 0,
            **kwargs
    ):
        super().__init__()

        attention_dim = encoder_output_size
        if use_embed:
            # only for count loss
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
                attention_dim, linear_units, nn.ReLU(), drop_prob,
                l1_kwargs=dict(
                    mode='lan', norm_fn=LayerNorm
                ),
                l2_kwargs=dict(
                    linear_fn=partial(nn.Linear, bias=False)
                )
            ),
            drop_prob,
        ) for _ in range(att_layer_num)])

        self.decoders3 = nn.ModuleList([SANMDecoderLayer(
            attention_dim,
            None,
            None,
            transformers.PositionWiseFeedForward(
                attention_dim, linear_units, nn.ReLU(), drop_prob,
                l1_kwargs=dict(
                    mode='lan', norm_fn=LayerNorm
                ),
                l2_kwargs=dict(
                    linear_fn=partial(nn.Linear, bias=False)
                )
            ),
            drop_prob,
        ) for _ in range(1)])

    def forward(
            self,
            hs_pad: torch.Tensor,
            hlens: torch.Tensor,
            ys_in_pad: torch.Tensor,
            ys_in_lens: torch.Tensor,
            chunk_mask: torch.Tensor = None,
    ):
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
            chunk_mask:
        Returns:
            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
        """
        tgt = ys_in_pad
        tgt_mask = attentions.make_pad_mask(ys_in_lens).to(device=tgt.device)

        memory = hs_pad
        memory_mask = attentions.make_pad_mask(hlens).to(device=memory.device)
        if chunk_mask is not None:
            memory_mask = memory_mask * chunk_mask
            if tgt_mask.size(1) != memory_mask.size(1):
                memory_mask = torch.cat((memory_mask, memory_mask[:, -2:-1, :]), dim=1)

        outputs = dict(
            tgt=tgt,
            tgt_mask=tgt_mask,
            context=memory,
            context_mask=memory_mask,
        )
        for m in self.decoders:
            outputs.update(m(**outputs))
        for m in self.decoders3:
            outputs.update(m(**outputs))
        x = outputs["tgt"]
        hidden = self.after_norm(x)
        x = self.output_layer(hidden)
        return dict(
            hidden=hidden,
            x=x
        )

    def forward_asf6(
            self,
            hs_pad: torch.Tensor,
            hlens: torch.Tensor,
            ys_in_pad: torch.Tensor,
            ys_in_lens: torch.Tensor,
    ):

        tgt = ys_in_pad
        tgt_mask = attentions.make_pad_mask(ys_in_lens).to(device=tgt.device)

        memory = hs_pad
        memory_mask = attentions.make_pad_mask(hlens).to(device=memory.device)

        outputs = dict(
            tgt=tgt,
            tgt_mask=tgt_mask,
            context=memory,
            context_mask=memory_mask,
        )
        for m in self.decoders[:5]:
            outputs.update(m(**outputs))

        attn_mat = self.decoders[5].get_attn_mat(**outputs)
        return attn_mat


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

    def forward(self, tgt, tgt_mask, context, context_mask=None, **kwargs):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor): Input tensor (#batch, maxlen_out, size).
            tgt_mask (torch.Tensor): Mask for input tensor (#batch, maxlen_out).
            context (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, size).
            context_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).

        Returns:
            torch.Tensor: Output tensor(#batch, maxlen_out, size).
            torch.Tensor: Mask for output tensor (#batch, maxlen_out).
            torch.Tensor: Encoded memory (#batch, maxlen_in, size).
            torch.Tensor: Encoded memory mask (#batch, maxlen_in).

        """
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        outputs = dict()
        x = tgt
        if self.self_attn is not None:
            tgt = self.norm2(tgt)
            x = self.self_attn(tgt, tgt_mask)
            x = residual + self.dropout(x)
            outputs['x_self_attn'] = x

        if self.src_attn is not None:
            residual = x
            x = self.norm3(x)
            x_src_attn = self.src_attn(x, context, context_mask)
            outputs['x_src_attn'] = x_src_attn
            x = residual + self.dropout(x_src_attn)

        outputs['tgt'] = x

        return outputs

    def get_attn_mat(self, tgt, tgt_mask, context, context_mask=None, **kwargs):
        residual = tgt
        tgt = self.norm1(tgt)
        tgt = self.feed_forward(tgt)

        x = tgt
        if self.self_attn is not None:
            tgt = self.norm2(tgt)
            x = self.self_attn(tgt, tgt_mask)
            x = residual + x

        x = self.norm3(x)
        q, k, v = self.src_attn.forward_in(x, context)
        scale = q.shape[-1] ** -0.5
        sim = torch.matmul(q, k.transpose(-2, -1)) * scale
        sim = attentions.mask_values(sim, context_mask, use_min=True)
        attn = F.softmax(sim, dim=-1)
        return attn


class SANMDecoderAttention(FSMNBlock):
    """actually it has no attention layers"""

    def __init__(self, model_dim, **kwargs):
        self.model_dim = model_dim
        super().__init__(**kwargs)

    def forward(self, x, attention_mask=None, mask_shift_chunk=None):
        return self.forward_fsmn(x, attention_mask, mask_shift_chunk)


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


class CifPredictorV2(nn.Module):
    """Continuous integrate-and-fire(Cif)
    code: `funasr.models.paraformer.cif_predictor.CifPredictorV2`
    without timestamps predicting"""
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
        self.dropout = nn.Dropout(self.drop_prob)

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

        context = hidden.transpose(1, 2)  # (b, s, d) -> (b, d, s)
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
            # note
            target_length = target_label_length.squeeze(-1)
        elif target_label is not None:
            target_length = (target_label != ignore_id).float().sum(-1)
        else:
            target_length = None
        token_num = alphas.sum(-1)

        if target_length is not None:
            alphas *= (target_length / token_num)[:, None].repeat(1, alphas.size(1))
        elif self.tail_threshold > 0.0:
            h, alphas, token_num = self.tail_process_fn(h, alphas, mask=mk)

        acoustic_embeds = self.cif(h, alphas)
        if target_length is None and self.tail_threshold > 0.0:
            token_num_int = torch.max(token_num).type(torch.int32).item()
            acoustic_embeds = acoustic_embeds[:, :token_num_int, :]

        return acoustic_embeds, token_num, alphas

    def tail_process_fn(self, hidden, alphas, mask=None):
        b, s, d = hidden.size()
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
        fires, fire_idxs = self.cif_wo_hidden(alphas)

        device = hidden.device
        dtype = hidden.dtype
        batch_size, len_time, hidden_size = hidden.size()
        prefix_sum_hidden = torch.cumsum(alphas.unsqueeze(-1).repeat((1, 1, hidden_size)) * hidden, dim=1)

        frames = prefix_sum_hidden[fire_idxs]
        shift_frames = torch.roll(frames, 1, dims=0)

        batch_len = fire_idxs.sum(1)
        batch_idxs = torch.cumsum(batch_len, dim=0)
        shift_batch_idxs = torch.roll(batch_idxs, 1, dims=0)
        shift_batch_idxs[0] = 0
        shift_frames[shift_batch_idxs] = 0

        remains = fires - torch.floor(fires)
        remain_frames = remains[fire_idxs].unsqueeze(-1).repeat((1, hidden_size)) * hidden[fire_idxs]

        shift_remain_frames = torch.roll(remain_frames, 1, dims=0)
        shift_remain_frames[shift_batch_idxs] = 0

        frames = frames - shift_frames + shift_remain_frames - remain_frames

        max_label_len = torch.round(alphas.sum(-1)).int().max()  # torch.round to calculate the max length

        frame_fires = torch.zeros(batch_size, max_label_len, hidden_size, dtype=dtype, device=device)
        indices = torch.arange(max_label_len, device=device).expand(batch_size, -1)
        frame_fires_idxs = indices < batch_len.unsqueeze(1)
        frame_fires[frame_fires_idxs] = frames
        return frame_fires

    def cif_wo_hidden(self, alphas):
        batch_size, len_time = alphas.size()
        device = alphas.device
        dtype = alphas.dtype
        fires = torch.zeros(batch_size, len_time, dtype=dtype, device=device)

        prefix_sum = torch.cumsum(alphas, dim=1, dtype=torch.float64).to(torch.float32)  # cumsum precision degradation cause wrong result in extreme
        prefix_sum_floor = torch.floor(prefix_sum)
        dislocation_prefix_sum = torch.roll(prefix_sum, 1, dims=1)
        dislocation_prefix_sum_floor = torch.floor(dislocation_prefix_sum)

        dislocation_prefix_sum_floor[:, 0] = 0
        dislocation_diff = prefix_sum_floor - dislocation_prefix_sum_floor

        fire_idxs = dislocation_diff > 0
        fires[fire_idxs] = 1
        fires = fires + prefix_sum - prefix_sum_floor
        return fires, fire_idxs


class CTC(nn.Module):
    """CTC module.

    Args:
        odim: dimension of outputs
        encoder_output_size: number of encoder projection units
        drop_prob: dropout prob (0.0 ~ 1.0)
        reduce: reduce the CTC loss into a scalar
    """

    def __init__(
            self,
            odim: int,
            encoder_output_size: int,
            drop_prob: float = 0.0,
            reduce: bool = True,
            ignore_nan_grad: bool = True,
    ):
        super().__init__()
        self.ctc_lo = nn.Linear(encoder_output_size, odim)
        self.ignore_nan_grad = ignore_nan_grad
        self.criterion = nn.CTCLoss(reduction="none")
        self.dropout = nn.Dropout(drop_prob)
        self.reduce = reduce

    def forward(self, hs_pad, hlens, ys_pad, ys_lens):
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(self.dropout(hs_pad))

        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        # (B, L) -> (BxL,)
        ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])

        hlens = hlens.to(hs_pad.device)
        loss = self.loss(ys_hat, ys_true, hlens, ys_lens).to(device=hs_pad.device, dtype=hs_pad.dtype)

        return loss

    def loss(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        th_pred = th_pred.log_softmax(2)
        loss = self.criterion(th_pred, th_target, th_ilen, th_olen)

        if loss.requires_grad and self.ignore_nan_grad:
            # ctc_grad: (L, B, O)
            ctc_grad = loss.grad_fn(torch.ones_like(loss))
            ctc_grad = ctc_grad.sum([0, 2])
            indices = torch.isfinite(ctc_grad)
            size = indices.long().sum()
            if size != th_pred.size(1):
                # Create mask for target
                target_mask = torch.full(
                    [th_target.size(0)],
                    1,
                    dtype=torch.bool,
                    device=th_target.device,
                )
                s = 0
                for ind, le in enumerate(th_olen):
                    if not indices[ind]:
                        target_mask[s: s + le] = 0
                    s += le

                # Calc loss again using maksed data
                loss = self.criterion(
                    th_pred[:, indices, :],
                    th_target[target_mask],
                    th_ilen[indices],
                    th_olen[indices],
                )
        else:
            size = th_pred.size(1)

        if self.reduce:
            # Batch-size average
            loss = loss.sum() / size
        else:
            loss = loss / size
        return loss
