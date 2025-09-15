import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..attentions import CrossAttention2D
from ..embeddings import LearnedPositionEmbedding
from ..layers import Linear, Residual


class EncoderEmbedding(nn.Module):
    """TokenEmbedding + PositionalEmbedding + SegmentEmbedding"""

    def __init__(self, vocab_size, embedding_dim, pad_id, max_seq_len=512, n_segment=2, drop_prob=0.1):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)

        # note, in vanilla attention, using cosine positional embeddings
        # to see `PositionalEmbedding` to get more detail
        # but in `transformers.BertForPreTraining`, using learned positional embeddings
        # to support weights from hf, there using learned positional embeddings also
        self.position = LearnedPositionEmbedding(max_seq_len, embedding_dim)

        # note, there add 1 to apply pad token usually
        # but in `transformers.BertForPreTraining` does not add yet
        # to support weights from hf, there do not add either
        self.segment = nn.Embedding(n_segment, embedding_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Dropout(drop_prob)
        )
        self.register_buffer("segment_label", torch.zeros(max_seq_len, dtype=torch.long), persistent=False)
        self.embedding_dim = embedding_dim
        self.pad_id = pad_id

    def forward(self, sequence, segment_label=None):
        """(b, s) -> (b, s, h)
        note, s is a dynamic var"""
        if segment_label is None:
            # use registered buffer to apply tracing the model
            b, s = sequence.shape[:2]
            segment_label = self.segment_label[:s]
            segment_label = segment_label.expand(b, s)
        x = (
                self.token(sequence)
                + self.position(sequence, self.pad_id)
                + self.segment(segment_label)
        )
        return self.head(x)


class DecoderEmbedding(nn.Module):
    """TokenEmbedding + PositionalEmbedding"""

    def __init__(self, vocab_size, embedding_dim, pad_id=None, max_seq_len=512):
        super().__init__()
        self.token = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.position = LearnedPositionEmbedding(max_seq_len, embedding_dim)

    def forward(self, sequence):
        """(b, s) -> (b, s, h)
        note, s is a dynamic var"""
        x = (
                self.token(sequence)
                + self.position(sequence)
        )
        return x


class TransformerSequential(nn.ModuleList):
    def __init__(self, *args, num_blocks=None, use_checkpoint=False, **kwargs):
        assert num_blocks is not None
        self.use_checkpoint = use_checkpoint
        super().__init__(
            [TransformerBlock(*args, **kwargs) for _ in range(num_blocks)]
        )

    def forward(self, x, attention_mask=None, callback_fn=None, per_block_kwargs=(), **block_kwargs):
        if self.training and self.use_checkpoint:
            # note, ensure having the right backward in checkpoint mode
            x.requires_grad_(True)
            # note, if having kwargs, use `use_reentrant=False`
            return checkpoint(self._forward, x, use_reentrant=False, attention_mask=attention_mask, callback_fn=callback_fn, per_block_kwargs=per_block_kwargs, **block_kwargs)
        else:
            return self._forward(x, attention_mask=attention_mask, callback_fn=callback_fn, per_block_kwargs=per_block_kwargs, **block_kwargs)

    def _forward(self, x, attention_mask=None, callback_fn=None, per_block_kwargs=(), **block_kwargs):
        for i, m in enumerate(self):
            if per_block_kwargs:
                block_kwargs.update(per_block_kwargs[i])
            x = m(x, attention_mask=attention_mask, **block_kwargs)
            if callback_fn:
                callback_fn(i, x)
        return x


class TransformerBlock(nn.Module):
    """SelfAttention + PositionWiseFeedForward
    refer to [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(
            self, hidden_size, num_attention_heads, ff_hidden_size,
            is_decode=False, norm_first=False, drop_prob=0.1,
            attention_fn=None, de_attention_fn=None,
            attend=None, de_attend=None, attend_fn=None, de_attend_fn=None,
            feed_forward_fn=None, norm_fn=None,
            fn_kwargs=dict(), de_fn_kwargs=dict(), ff_kwargs=dict(),
            attend_fn_kwargs=dict(), de_attend_fn_kwargs=dict(), norm_kwargs=dict(),
    ):
        super().__init__()
        attention_fn = attention_fn or CrossAttention2D
        de_attention_fn = de_attention_fn or CrossAttention2D
        norm_fn = norm_fn or nn.LayerNorm
        feed_forward_fn = feed_forward_fn or PositionWiseFeedForward

        if not attend and attend_fn:
            attend = attend_fn(**attend_fn_kwargs)
        if not de_attend and de_attend_fn:
            de_attend = de_attend_fn(**de_attend_fn_kwargs)

        attention_kwargs = dict(
            n_heads=num_attention_heads,
            model_dim=hidden_size,
            drop_prob=drop_prob,
        )
        fn_kwargs = {
            **attention_kwargs,
            **fn_kwargs
        }
        de_fn_kwargs = {
            **attention_kwargs,
            **de_fn_kwargs
        }
        ff_kwargs = {
            'drop_prob': drop_prob,
            **ff_kwargs
        }

        self.attn_res = Residual(
            attention_fn(attend=attend, **fn_kwargs),  # SelfAttention
            norm=norm_fn(hidden_size, **norm_kwargs),
            norm_first=norm_first
        )

        self.is_decode = is_decode
        if is_decode:
            self.de_attn_res = Residual(
                de_attention_fn(attend=de_attend, **de_fn_kwargs),  # CrossAttention
                norm=norm_fn(hidden_size, **norm_kwargs),
                norm_first=norm_first
            )

        self.ff_res = Residual(
            feed_forward_fn(hidden_size, ff_hidden_size, **ff_kwargs),
            norm=norm_fn(hidden_size, **norm_kwargs),
            norm_first=norm_first
        )

    def forward(self, x, context=None, attention_mask=None, context_mask=None, **attn_kwargs):
        """(b, s, h) -> (b, s, h)"""
        x = self.attn_res(x, attention_mask=attention_mask, **attn_kwargs)
        if self.is_decode:
            x = self.de_attn_res(x, k=context, v=context, attention_mask=context_mask, **attn_kwargs)
        x = self.ff_res(x)
        return x


class PositionWiseFeedForward(nn.Sequential):
    """y = F2(a(F1(x)))"""

    def __init__(self, hidden_size, ff_hidden_size, act=None, drop_prob=0.1, l1_kwargs=dict(), l2_kwargs=dict(), **kwargs):
        act = act or nn.GELU()
        _l1_kwargs = dict(
            mode='la', act=act
        )
        _l1_kwargs.update(kwargs)
        _l1_kwargs.update(l1_kwargs)

        _l2_kwargs = dict(
            mode='ld', drop_prob=drop_prob
        )
        _l2_kwargs.update(kwargs)
        _l2_kwargs.update(l2_kwargs)
        super().__init__(
            Linear(hidden_size, ff_hidden_size, **_l1_kwargs),
            Linear(ff_hidden_size, hidden_size, **_l2_kwargs)
        )
