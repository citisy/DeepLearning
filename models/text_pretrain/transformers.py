import math
import torch
import torch.nn.functional as F
from torch import nn
from utils import torch_utils
from ..layers import Linear, Residual
from ..attentions import CrossAttention2D


class TransformerBlock(nn.Module):
    """SelfAttention + PositionWiseFeedForward
    refer to [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
    """

    def __init__(
            self, hidden_size, num_attention_heads, feed_forward_hidden,
            is_decode=False, norm_first=False, norm_fn=None, drop_prob=0.1,
            attend=None, de_attend=None,
            fn_kwargs=dict(), de_fn_kwargs=dict(), ff_kwargs=dict(),
    ):
        super().__init__()
        norm_fn = norm_fn or nn.LayerNorm
        self.res1 = Residual(
            CrossAttention2D(n_heads=num_attention_heads, model_dim=hidden_size, drop_prob=drop_prob, attend=attend, **fn_kwargs),  # SelfAttention
            norm=norm_fn(hidden_size),
            norm_first=norm_first
        )

        self.is_decode = is_decode
        if is_decode:
            self.de_attn_res = Residual(
                CrossAttention2D(n_heads=num_attention_heads, model_dim=hidden_size, drop_prob=drop_prob, attend=de_attend, **de_fn_kwargs),  # CrossAttention
                norm=norm_fn(hidden_size),
                norm_first=norm_first
            )

        self.res2 = Residual(
            PositionWiseFeedForward(hidden_size, feed_forward_hidden, drop_prob=drop_prob, **ff_kwargs),  # PositionWiseFeedForward
            norm=norm_fn(hidden_size),
            norm_first=norm_first
        )

    def forward(self, x, context=None, attention_mask=None, context_mask=None, **kwargs):
        """(b, s, h) -> (b, s, h)"""
        x = self.res1(x, attention_mask=attention_mask, **kwargs)
        if self.is_decode:
            x = self.de_attn_res(x, k=context, v=context, attention_mask=context_mask, **kwargs)
        x = self.res2(x)
        return x


class PositionWiseFeedForward(nn.Sequential):
    def __init__(self, hidden_size, feed_forward_hidden, act=None, drop_prob=0.1, **kwargs):
        act = act or nn.GELU()
        super().__init__(
            Linear(hidden_size, feed_forward_hidden, mode='la', act=act, **kwargs),
            Linear(feed_forward_hidden, hidden_size, mode='ld', drop_prob=drop_prob, **kwargs)
        )
