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

    def __init__(self, hidden_size, num_attention_heads, feed_forward_hidden, norm_first=False, drop_prob=0.1, **attn_kwargs):
        super().__init__()
        self.res1 = Residual(
            CrossAttention2D(n_heads=num_attention_heads, model_dim=hidden_size, drop_prob=drop_prob, **attn_kwargs),   # SelfAttention
            norm=nn.LayerNorm(hidden_size),
            norm_first=norm_first
        )

        self.res2 = Residual(
            PositionWiseFeedForward(hidden_size, feed_forward_hidden, drop_prob),  # PositionWiseFeedForward
            norm=nn.LayerNorm(hidden_size),
            norm_first=norm_first
        )

    def forward(self, x, attention_mask=None):
        """(b, s, h) -> (b, s, h)"""
        x = self.res1(x, attention_mask=attention_mask)
        x = self.res2(x)
        return x


class PositionWiseFeedForward(nn.Sequential):
    def __init__(self, hidden_size, feed_forward_hidden, drop_prob=0.1):
        super().__init__(
            Linear(hidden_size, feed_forward_hidden, mode='la', act=nn.GELU()),
            Linear(feed_forward_hidden, hidden_size, mode='ld', drop_prob=drop_prob)
        )
