import torch
from torch import nn
from ..layers import Linear, Residual
from ..attentions import CrossAttention2D
from ..embeddings import LearnedPositionEmbedding


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
        self.embedding_dim = embedding_dim

    def forward(self, sequence, segment_label):
        """(b, s) -> (b, s, h)
        note, s is a dynamic var"""
        x = (
                self.token(sequence)
                + self.position(sequence)
                + self.segment(segment_label)
        )
        return self.head(x)


class DecoderEmbedding(nn.Module):
    """TokenEmbedding + PositionalEmbedding + SegmentEmbedding"""

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
    def __init__(self, *args, num_blocks=None, **kwargs):
        assert num_blocks is not None
        super().__init__(
            [TransformerBlock(*args, **kwargs) for _ in range(num_blocks)]
        )

    def forward(self, x, attention_mask=None, callback_fn=None, **kwargs):
        for i, m in enumerate(self):
            x = m(x, attention_mask=attention_mask, **kwargs)
            if callback_fn:
                callback_fn(i, x)
        return x


class TransformerBlock(nn.Module):
    """SelfAttention + PositionWiseFeedForward
    refer to [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(
            self, hidden_size, num_attention_heads, ff_hidden_size,
            is_decode=False, norm_first=False, separate=True, drop_prob=0.1,
            attend=None, de_attend=None, attend_fn=None, de_attend_fn=None,
            feed_forward_fn=None, norm_fn=None,
            fn_kwargs=dict(), de_fn_kwargs=dict(), ff_kwargs=dict(),
            attend_fn_kwargs=dict(), de_attend_fn_kwargs=dict(), norm_kwargs=dict(),
    ):
        super().__init__()
        norm_fn = norm_fn or nn.LayerNorm
        feed_forward_fn = feed_forward_fn or PositionWiseFeedForward
        if not attend and attend_fn:
            attend = attend_fn(**attend_fn_kwargs)
        if not de_attend and de_attend_fn:
            de_attend = de_attend_fn(**de_attend_fn_kwargs)

        self.attn_res = Residual(
            CrossAttention2D(n_heads=num_attention_heads, model_dim=hidden_size, drop_prob=drop_prob, attend=attend, separate=separate, **fn_kwargs),  # SelfAttention
            norm=norm_fn(hidden_size, **norm_kwargs),
            norm_first=norm_first
        )

        self.is_decode = is_decode
        if is_decode:
            self.de_attn_res = Residual(
                CrossAttention2D(n_heads=num_attention_heads, model_dim=hidden_size, drop_prob=drop_prob, attend=de_attend, separate=separate, **de_fn_kwargs),  # CrossAttention
                norm=norm_fn(hidden_size),
                norm_first=norm_first
            )

        self.ff_res = Residual(
            feed_forward_fn(hidden_size, ff_hidden_size, drop_prob=drop_prob, **ff_kwargs),
            norm=norm_fn(hidden_size),
            norm_first=norm_first
        )

    def forward(self, x, context=None, attention_mask=None, context_mask=None, **kwargs):
        """(b, s, h) -> (b, s, h)"""
        x = self.attn_res(x, attention_mask=attention_mask, **kwargs)
        if self.is_decode:
            x = self.de_attn_res(x, k=context, v=context, attention_mask=context_mask, **kwargs)
        x = self.ff_res(x)
        return x


class PositionWiseFeedForward(nn.Sequential):
    """y = F2(a(F1(x)))"""

    def __init__(self, hidden_size, feed_forward_hidden, act=None, drop_prob=0.1, **kwargs):
        act = act or nn.GELU()
        super().__init__(
            Linear(hidden_size, feed_forward_hidden, mode='la', act=act, **kwargs),
            Linear(feed_forward_hidden, hidden_size, mode='ld', drop_prob=drop_prob, **kwargs)
        )


def make_causal_attention_mask(x, start_pos=0):
    """
    e.g.:
        x.shape=(b, 3, -1) -> mask.shape=(b, 1, 3, 3)
        [[1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]

    """
    batch_size, seq_len = x.shape[:2]
    mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
    mask = torch.hstack([
        torch.ones((seq_len, start_pos), dtype=mask.dtype, device=x.device),
        mask
    ])
    mask = mask[None, None].repeat(batch_size, 1, 1, 1)  # (b, 1, s, s+p)
    return mask


class EmbeddingSim(nn.Module):
    """as a linear layer"""

    def __init__(self, weight, use_bias=True):
        super().__init__()
        self.weight = weight
        self.use_bias = use_bias
        if use_bias:
            num_embeddings = weight.shape[0]
            self.bias = nn.Parameter(torch.zeros(num_embeddings))

    def forward(self, x):
        y = x.matmul(self.weight.transpose(1, 0).detach())
        if self.use_bias:
            y += self.bias
        return y
