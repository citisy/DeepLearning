from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from . import Paraformer
from .Paraformer import WeightConverter
from ..bundles import WeightLoader
from .. import attentions
from ..text_pretrain import transformers
from torch.nn.utils.rnn import pad_sequence


class Model(Paraformer.Model):
    """refer to:
    paper:
        - [FunASR: A Fundamental End-to-End Speech Recognition Toolkit](https://arxiv.org/abs/2305.11013)
    """

    def make_decoder(self, inner_dim=512, bias_encoder_dropout_prob=0., **decoder_config):
        self.bias_encoder = nn.LSTM(
            inner_dim, inner_dim, 1, batch_first=True, dropout=bias_encoder_dropout_prob
        )
        self.bias_embed = nn.Embedding(self.vocab_size, inner_dim)
        self.decoder = ContextualParaformerDecoder(
            vocab_size=self.vocab_size,
            encoder_output_size=self.encoder.output_size,
            **decoder_config,
        )

    def post_process(self, encoder_out, encoder_out_mask, speech_lens, hotword_ids=None, clas_scale=1.0, **kwargs):
        # predict the length of token
        pre_acoustic_embeds, token_num, _ = self.neck(encoder_out, None, encoder_out_mask, ignore_id=self.ignore_id)

        token_num = token_num.round().long()
        if torch.max(token_num) < 1:
            return {}

        decoder_out = self.decode(
            encoder_out, speech_lens, pre_acoustic_embeds, token_num,
            hotword_ids=hotword_ids,
            clas_scale=clas_scale,
        )

        preds = []
        b, n, d = decoder_out.size()
        for i in range(b):
            am_scores = decoder_out[i, : token_num[i], :]
            pred = am_scores.argmax(dim=-1)
            keep = (pred != self.blank_id) & (pred != self.sos_id) & (pred != self.eos_id)
            pred = pred[keep]
            preds.append(pred)

        # note, no timestamp output
        return dict(
            preds=preds,
        )

    def decode(self, encoder_out, encoder_out_lens, sematic_embeds, ys_pad_lens, hotword_ids=None, clas_scale=1.0):
        if hotword_ids is None:
            hotword_ids = [torch.Tensor([1]).long().to(encoder_out.device)]  # empty hotword list
            hw_list_pad = pad_sequence(hotword_ids, batch_first=True)

            hw_embed = self.bias_embed(hw_list_pad)
            hw_embed, (h_n, _) = self.bias_encoder(hw_embed)
            hw_embed = h_n.repeat(encoder_out.shape[0], 1, 1)
        else:
            hw_lengths = [len(i) for i in hotword_ids]
            hw_list_pad = pad_sequence([torch.Tensor(i).long() for i in hotword_ids], batch_first=True).to(encoder_out.device)
            hw_embed = self.bias_embed(hw_list_pad)
            hw_embed = nn.utils.rnn.pack_padded_sequence(hw_embed, hw_lengths, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.bias_encoder(hw_embed)
            hw_embed = h_n.repeat(encoder_out.shape[0], 1, 1)

        decoder_out = self.decoder(
            encoder_out,
            encoder_out_lens,
            sematic_embeds,
            ys_pad_lens,
            contextual_info=hw_embed,
            clas_scale=clas_scale,
        )

        decoder_out = F.log_softmax(decoder_out, dim=-1)
        return decoder_out


class ContextualParaformerDecoder(nn.Module):
    """
    code: `funasr.models.contextual_paraformer.decoder.ContextualParaformerDecoder`
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

        self.after_norm = Paraformer.LayerNorm(attention_dim)
        self.output_layer = nn.Linear(attention_dim, vocab_size) if use_output_layer else nn.Identity()

        self.att_layer_num = att_layer_num
        self.num_blocks = num_blocks
        if sanm_shfit is None:
            sanm_shfit = (kernel_size - 1) // 2
        self.decoders = nn.ModuleList([Paraformer.SANMDecoderLayer(
            attention_dim,
            Paraformer.SANMDecoderAttention(
                model_dim=attention_dim,
                kernel_size=kernel_size,
                sanm_shfit=sanm_shfit
            ),
            Paraformer.CrossAttention2D(
                attention_heads,
                attention_dim,
                src_attention_dropout_rate,
            ),
            transformers.PositionWiseFeedForward(
                attention_dim, linear_units, nn.ReLU(), drop_prob,
                l1_kwargs=dict(
                    mode='lan', norm_fn=Paraformer.LayerNorm
                ),
                l2_kwargs=dict(
                    linear_fn=partial(nn.Linear, bias=False)
                )
            ),
            drop_prob,
        ) for _ in range(att_layer_num - 1)])

        # note, split the last_decoder from decoders
        self.last_decoder = Paraformer.SANMDecoderLayer(
            attention_dim,
            Paraformer.SANMDecoderAttention(
                model_dim=attention_dim,
                kernel_size=kernel_size,
                sanm_shfit=sanm_shfit
            ),
            Paraformer.CrossAttention2D(
                attention_heads,
                attention_dim,
                src_attention_dropout_rate,
            ),
            transformers.PositionWiseFeedForward(
                attention_dim, linear_units, nn.ReLU(), drop_prob,
                l1_kwargs=dict(
                    mode='lan', norm_fn=Paraformer.LayerNorm
                ),
                l2_kwargs=dict(
                    linear_fn=partial(nn.Linear, bias=False)
                )
            ),
            drop_prob,
        )

        self.bias_decoder = ContextualBiasDecoder(
            size=attention_dim,
            src_attn=Paraformer.CrossAttention2D(
                attention_heads,
                attention_dim,
                src_attention_dropout_rate,
            ),
            dropout_rate=drop_prob,
            normalize_before=True,
        )
        self.bias_output = nn.Conv1d(attention_dim * 2, attention_dim, 1, bias=False)
        self.dropout = torch.nn.Dropout(drop_prob)

        self.decoders3 = nn.ModuleList([Paraformer.SANMDecoderLayer(
            attention_dim,
            None,
            None,
            transformers.PositionWiseFeedForward(
                attention_dim, linear_units, nn.ReLU(), drop_prob,
                l1_kwargs=dict(
                    mode='lan', norm_fn=Paraformer.LayerNorm
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
            contextual_info: torch.Tensor = None,
            clas_scale: float = 1.0,
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

        outputs.update(self.last_decoder(**outputs))
        x_self_attn = outputs['x_self_attn']
        x_src_attn = outputs['x_src_attn']

        # contextual paraformer related
        contextual_length = torch.Tensor([contextual_info.shape[1]]).int().repeat(hs_pad.shape[0])
        contextual_mask = self.sequence_mask(contextual_length, device=memory.device)[:, None, :]
        cx = self.bias_decoder(x_self_attn, contextual_info, memory_mask=contextual_mask)

        if self.bias_output is not None:
            x = torch.cat([x_src_attn, cx * clas_scale], dim=2)
            x = self.bias_output(x.transpose(1, 2)).transpose(1, 2)  # 2D -> D
            x = x_self_attn + self.dropout(x)
            outputs.update(
                tgt=x,
            )

        for m in self.decoders3:
            outputs.update(m(**outputs))
        x = outputs["tgt"]
        hidden = self.after_norm(x)
        x = self.output_layer(hidden)
        return x

    @staticmethod
    def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix
        mask = mask.detach()

        return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class ContextualBiasDecoder(nn.Module):
    def __init__(
            self,
            size,
            src_attn,
            dropout_rate,
            normalize_before=True,
    ):
        """Construct an DecoderLayer object."""
        super().__init__()
        self.size = size
        self.src_attn = src_attn
        if src_attn is not None:
            self.norm3 = Paraformer.LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(self, tgt, memory, memory_mask=None, **kwargs):
        x = tgt
        if self.src_attn is not None:
            if self.normalize_before:
                x = self.norm3(x)
            x = self.dropout(self.src_attn(x, memory, memory_mask))
        return x
