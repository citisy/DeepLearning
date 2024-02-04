import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from functools import partial
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce
from .layers import Linear, Conv


def get_attention_input(n_heads=None, model_dim=None, head_dim=None):
    if n_heads and model_dim:
        assert model_dim % n_heads == 0
        head_dim = model_dim // n_heads
    elif n_heads and head_dim:
        model_dim = n_heads * head_dim
    elif model_dim and head_dim:
        assert model_dim % head_dim == 0
        n_heads = model_dim // head_dim
    else:
        raise ValueError('Must set two of [n_heads, model_dim, head_dim] at the same time')

    return n_heads, model_dim, head_dim


class CrossAttention2D(nn.Module):
    """cross attention"""

    def __init__(self, n_heads=None, model_dim=None, head_dim=None, query_dim=None, context_dim=None,
                 use_xformers=None, use_conv=False, separate_conv=True, use_mem_kv=False, n_mem_size=4,
                 drop_prob=0.1, **fn_kwargs):
        super().__init__()
        n_heads, model_dim, head_dim = get_attention_input(n_heads, model_dim, head_dim)
        query_dim = query_dim or model_dim
        context_dim = context_dim or model_dim

        self.use_conv = use_conv
        self.separate_conv = separate_conv
        self.use_mem_kv = use_mem_kv

        if use_conv:  # build by conv func
            if separate_conv:
                self.to_qkv = nn.ModuleList([
                    nn.Conv1d(query_dim, model_dim, 1, **fn_kwargs),
                    nn.Conv1d(context_dim, model_dim, 1, **fn_kwargs),
                    nn.Conv1d(context_dim, model_dim, 1, **fn_kwargs),
                ])
            else:  # only for self attention
                assert query_dim == context_dim

                # different to linear function, each conv filter and feature map is independent
                # so can use a conv layer to compute, and then, chunk it
                self.to_qkv = nn.Conv1d(query_dim, model_dim * 3, 1, **fn_kwargs)

            self.view_in = Rearrange('b (n c) dk-> b n c dk', n=n_heads)

            if use_mem_kv:
                self.mem_kv = nn.Parameter(torch.randn(2, n_heads, n_mem_size, head_dim))

            self.view_out = Rearrange('b n c dk -> b (n c) dk')
            self.to_out = nn.Conv1d(model_dim, query_dim, **fn_kwargs)

        else:  # build by linear func
            self.to_qkv = nn.ModuleList([
                nn.Linear(query_dim, model_dim, **fn_kwargs),
                nn.Linear(context_dim, model_dim, **fn_kwargs),
                nn.Linear(context_dim, model_dim, **fn_kwargs),
            ])
            self.view_in = Rearrange('b s (n dk)-> b n s dk', n=n_heads)

            self.view_out = Rearrange('b n s dk -> b s (n dk)')
            self.to_out = Linear(model_dim, query_dim, mode='ld', drop_prob=drop_prob)

        if use_xformers:
            # faster and less memory
            # requires pytorch > 2.0
            import xformers
            self.attend = xformers.ops.memory_efficient_attention
        else:
            self.attend = ScaleAttend(drop_prob)

    def forward(self, q, k=None, v=None, attention_mask=None):
        if self.use_conv and self.separate_conv:  # note, only for self attention
            q, k, v = self.to_qkv(q).chunk(3, dim=1)
        else:
            if k is None:
                k = q
            if v is None:
                v = q

            q, k, v = [m(x) for m, x in zip(self.to_qkv, (q, k, v))]

        q, k, v = [self.view_in(x) for x in (q, k, v)]

        if self.use_conv and self.use_mem_kv:
            mk, mv = map(lambda t: repeat(t, 'n j d -> b n j d', b=q.shape[0]), self.mem_kv)
            k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        x = self.attend(q, k, v, attention_mask=attention_mask)
        x = self.view_out(x)

        return self.to_out(x)


class CrossAttention3D(nn.Module):
    """cross attention build by conv function"""

    def __init__(self, n_heads=None, model_dim=None, head_dim=None, query_dim=None, context_dim=None,
                 use_xformers=None, separate_conv=True, use_mem_kv=False, n_mem_size=4,
                 drop_prob=0., **conv_kwargs):
        super().__init__()
        n_heads, model_dim, head_dim = get_attention_input(n_heads, model_dim, head_dim)
        query_dim = query_dim or model_dim
        context_dim = context_dim or model_dim
        self.scale = head_dim ** -0.5
        self.n_heads = n_heads
        self.use_mem_kv = use_mem_kv
        self.separate_conv = separate_conv

        if separate_conv:
            self.to_qkv = nn.ModuleList([
                nn.Conv2d(query_dim, model_dim, 1, **conv_kwargs),
                nn.Conv2d(context_dim, model_dim, 1, **conv_kwargs),
                nn.Conv2d(context_dim, model_dim, 1, **conv_kwargs),
            ])
        else:  # only for self attention
            assert query_dim == context_dim

            # different to linear function, each conv filter and feature map is independent
            # so can use a conv layer to compute, and then, chunk it
            self.to_qkv = nn.Conv2d(query_dim, model_dim * 3, 1, **conv_kwargs)

        if use_mem_kv:
            self.view_in = Rearrange('b (n d) h w -> b n (h w) d', n=n_heads)
            self.mem_kv = nn.Parameter(torch.randn(2, n_heads, n_mem_size, head_dim))
        else:
            self.view_in = Rearrange('b c h w -> b (h w) c')

        if use_xformers:
            # faster and less memory
            # requires pytorch > 2.0
            import xformers
            self.attend = xformers.ops.memory_efficient_attention
        else:
            self.attend = ScaleAttend(drop_prob)
        self.to_out = nn.Conv2d(model_dim, query_dim, 1)

    def forward(self, q, k=None, v=None):
        b, c, h, w = q.shape
        if self.separate_conv:
            if k is None:
                k = q
            if v is None:
                v = q

            q, k, v = [m(x) for m, x in zip(self.to_qkv, (q, k, v))]
        else:  # only for self attention
            q, k, v = self.to_qkv(q).chunk(3, dim=1)

        q, k, v = [self.view_in(x) for x in (q, k, v)]

        if self.use_mem_kv:
            mk, mv = map(lambda t: repeat(t, 'n j d -> b n j d', b=b), self.mem_kv)
            k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        x = self.attend(q, k, v)
        if self.use_mem_kv:
            x = rearrange(x, 'b n (h w) d -> b (n d) h w', h=h, w=w)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # view_out
        return self.to_out(x)


class ScaleAttend(nn.Module):
    """Scaled Dot-Product Attention
    attn(q, k, v) = softmax(qk'/sqrt(dk))*v"""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, q, k, v, attention_mask=None):
        scale = q.shape[-1] ** -0.5
        # similarity
        # sim = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scale
        sim = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:  # mask pad
            attention_mask = ~attention_mask.to(dtype=torch.bool)
            attention_mask = attention_mask[:, None, None].repeat(1, 1, sim.size(2), 1)  # mask pad
            sim = sim.masked_fill(attention_mask, torch.finfo(sim.dtype).min)  # support fp16

        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)

        # attn = einsum('... i j, ... j d -> ... i d', attn, v)
        attn = torch.matmul(attn, v)

        return attn


class LinearAttention3D(nn.Module):
    """linear attention build by conv function"""

    def __init__(self, n_heads=None, model_dim=None, head_dim=None, query_dim=None, context_dim=None,
                 separate_conv=True, use_mem_kv=True, n_mem_size=4, norm=None, **conv_kwargs):
        super().__init__()
        n_heads, model_dim, head_dim = get_attention_input(n_heads, model_dim, head_dim)
        query_dim = query_dim or model_dim
        context_dim = context_dim or model_dim
        self.scale = head_dim ** -0.5
        self.n_heads = n_heads
        self.use_mem_kv = use_mem_kv
        self.separate_conv = separate_conv

        if separate_conv:
            self.to_qkv = nn.ModuleList([
                nn.Conv2d(query_dim, model_dim, 1, **conv_kwargs),
                nn.Conv2d(context_dim, model_dim, 1, **conv_kwargs),
                nn.Conv2d(context_dim, model_dim, 1, **conv_kwargs),
            ])
        else:  # only for self attention
            assert query_dim == context_dim

            # different to linear function, each conv filter and feature map is independent
            # so can use a conv layer to compute, and then, chunk it
            self.to_qkv = nn.Conv2d(query_dim, model_dim * 3, 1, **conv_kwargs)

        self.cvt = Rearrange('b (n d) h w -> b n d (h w)', n=n_heads)

        if use_mem_kv:
            self.mem_kv = nn.Parameter(torch.randn(2, n_heads, head_dim, n_mem_size))

        self.attend = LinearAttend()
        self.to_out = Conv(model_dim, query_dim, 1, mode='cn', norm=norm)

    def forward(self, q, k=None, v=None):
        b, c, h, w = q.shape
        if self.separate_conv:
            if k is None:
                k = q
            if v is None:
                v = q

            q, k, v = [m(x) for m, x in zip(self.to_qkv, (q, k, v))]
        else:  # only for self attention
            q, k, v = self.to_qkv(q).chunk(3, dim=1)

        q, k, v = [self.cvt(x) for x in (q, k, v)]

        if self.use_mem_kv:
            mk, mv = map(lambda t: repeat(t, 'n d j -> b n d j', b=b), self.mem_kv)
            k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        x = self.attend(q, k, v)
        x = rearrange(x, 'b n d (h w) -> b (n d) h w', n=self.n_heads, h=h, w=w)
        return self.to_out(x)


class LinearAttend(nn.Module):
    """linear attention, to reduce the computation
    refer to: https://arxiv.org/pdf/2006.16236.pdf
    attn(q, k, v) = softmax(k)*v*softmax(q)/sqrt(dk)
    """

    def __init__(self, drop_prob=0.):
        super().__init__()

    def attend(self, q, k, v):
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        scale = q.shape[-2] ** -0.5
        q = q * scale

        context = torch.einsum('b n i s, b n j s -> b n i j', k, v)  # d = i = j
        context = torch.einsum('b n i j, b n i s -> b n j s', context, q)
        return context


class FlashAttend(nn.Module):
    """requires pytorch > 2.0
    refer to: https://arxiv.org/pdf/2205.14135.pdf"""

    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

        # determine efficient attention configs for cuda and cpu
        self.cpu_config = dict(
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True
        )

        self.cuda_config = None

        if not torch.cuda.is_available():
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))
        if device_properties.major == 8 and device_properties.minor == 0:
            self.cuda_config = dict(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False
            )
        else:
            self.cuda_config = dict(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=True
            )

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        _, heads, q_len, _ = q.shape

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention
        config = self.cuda_config if q.is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.drop_prob if self.training else 0.
            )

        return out
