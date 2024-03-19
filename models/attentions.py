import math
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


def get_qkv(q, k=None, v=None):
    k = q if k is None else k
    v = q if v is None else v
    return q, k, v


class CrossAttention2D(nn.Module):
    """cross attention"""

    def __init__(self, n_heads=None, model_dim=None, head_dim=None, query_dim=None, context_dim=None,
                 use_conv=False, separate=True, drop_prob=0.1, attend=None, out_fn=None, **fn_kwargs):
        super().__init__()
        n_heads, model_dim, head_dim = get_attention_input(n_heads, model_dim, head_dim)
        query_dim = query_dim or model_dim
        context_dim = context_dim or model_dim

        self.use_conv = use_conv
        self.separate = separate

        if use_conv:  # build by conv func
            if separate:
                self.to_qkv = nn.ModuleList([
                    nn.Conv1d(query_dim, model_dim, 1, **fn_kwargs),
                    nn.Conv1d(context_dim, model_dim, 1, **fn_kwargs),
                    nn.Conv1d(context_dim, model_dim, 1, **fn_kwargs),
                ])
            else:  # only for self attention
                assert query_dim == context_dim
                self.to_qkv = nn.Conv1d(query_dim, model_dim * 3, 1, **fn_kwargs)

            self.view_in = Rearrange('b (n c) dk-> b n c dk', n=n_heads)

            self.view_out = Rearrange('b n c dk -> b (n c) dk')
            self.to_out = nn.Conv1d(model_dim, query_dim, **fn_kwargs) if out_fn is None else out_fn

        else:  # build by linear func
            if separate:
                self.to_qkv = nn.ModuleList([
                    nn.Linear(query_dim, model_dim, **fn_kwargs),
                    nn.Linear(context_dim, model_dim, **fn_kwargs),
                    nn.Linear(context_dim, model_dim, **fn_kwargs),
                ])
            else:  # only for self attention
                assert query_dim == context_dim
                self.to_qkv = nn.Linear(query_dim, model_dim * 3, **fn_kwargs)

            self.view_in = Rearrange('b s (n dk)-> b n s dk', n=n_heads)

            self.view_out = Rearrange('b n s dk -> b s (n dk)')
            self.to_out = Linear(model_dim, query_dim, mode='ld', drop_prob=drop_prob, **fn_kwargs) if out_fn is None else out_fn

        self.attend = ScaleAttend(drop_prob=drop_prob) if attend is None else attend

    def forward(self, q, k=None, v=None, attention_mask=None, **kwargs):
        if self.separate:
            q, k, v = get_qkv(q, k, v)
            q, k, v = [m(x) for m, x in zip(self.to_qkv, (q, k, v))]
        else:
            dim = 1 if self.use_conv else -1
            q, k, v = self.to_qkv(q).chunk(3, dim=dim)

        q, k, v = [self.view_in(x).contiguous() for x in (q, k, v)]

        x = self.attend(q, k, v, attention_mask=attention_mask, **kwargs)
        x = self.view_out(x)

        x = self.to_out(x)
        return x


class CrossAttention3D(nn.Module):
    """cross attention build by conv function"""

    def __init__(self, n_heads=None, model_dim=None, head_dim=None, query_dim=None, context_dim=None,
                 separate=True, drop_prob=0., attend=None, out_fn=None, **fn_kwargs):
        super().__init__()
        n_heads, model_dim, head_dim = get_attention_input(n_heads, model_dim, head_dim)
        query_dim = query_dim or model_dim
        context_dim = context_dim or model_dim
        self.scale = head_dim ** -0.5
        self.n_heads = n_heads
        self.separate = separate

        if separate:
            self.to_qkv = nn.ModuleList([
                nn.Conv2d(query_dim, model_dim, 1, **fn_kwargs),
                nn.Conv2d(context_dim, model_dim, 1, **fn_kwargs),
                nn.Conv2d(context_dim, model_dim, 1, **fn_kwargs),
            ])
        else:  # only for self attention
            assert query_dim == context_dim
            self.to_qkv = nn.Conv2d(query_dim, model_dim * 3, 1, **fn_kwargs)

        self.view_in = Rearrange('b c h w -> b 1 (h w) c')
        self.attend = ScaleAttend(drop_prob=drop_prob) if attend is None else attend
        self.view_out = partial(rearrange, pattern='b 1 (h w) c -> b c h w')  # view_out)
        self.to_out = nn.Conv2d(model_dim, query_dim, 1) if out_fn is None else out_fn

    def forward(self, q, k=None, v=None):
        b, c, h, w = q.shape
        if self.separate:
            q, k, v = get_qkv(q, k, v)
            q, k, v = [m(x) for m, x in zip(self.to_qkv, (q, k, v))]
        else:  # only for self attention
            q, k, v = self.to_qkv(q).chunk(3, dim=1)

        q, k, v = [self.view_in(x).contiguous() for x in (q, k, v)]

        x = self.attend(q, k, v)
        x = self.view_out(x, h=h, w=w)
        return self.to_out(x)


class LinearAttention3D(nn.Module):
    """linear attention build by conv function"""

    def __init__(self, n_heads=None, model_dim=None, head_dim=None, query_dim=None, context_dim=None,
                 separate=True, attend=None, out_fn=None, **fn_kwargs):
        super().__init__()
        n_heads, model_dim, head_dim = get_attention_input(n_heads, model_dim, head_dim)
        query_dim = query_dim or model_dim
        context_dim = context_dim or model_dim
        self.scale = head_dim ** -0.5
        self.n_heads = n_heads
        self.separate = separate

        if separate:
            self.to_qkv = nn.ModuleList([
                nn.Conv2d(query_dim, model_dim, 1, **fn_kwargs),
                nn.Conv2d(context_dim, model_dim, 1, **fn_kwargs),
                nn.Conv2d(context_dim, model_dim, 1, **fn_kwargs),
            ])
        else:  # only for self attention
            assert query_dim == context_dim
            self.to_qkv = nn.Conv2d(query_dim, model_dim * 3, 1, **fn_kwargs)

        self.view_in = Rearrange('b (n d) h w -> b n d (h w)', n=n_heads)
        self.attend = LinearAttend() if attend is None else attend
        self.view_out = partial(rearrange, pattern='b n d (h w) -> b (n d) h w')
        self.to_out = nn.Conv2d(model_dim, query_dim, 1) if out_fn is None else out_fn

    def forward(self, q, k=None, v=None):
        b, c, h, w = q.shape
        if self.separate:
            q, k, v = get_qkv(q, k, v)
            q, k, v = [m(x) for m, x in zip(self.to_qkv, (q, k, v))]
        else:  # only for self attention
            q, k, v = self.to_qkv(q).chunk(3, dim=1)

        q, k, v = [self.view_in(x) for x in (q, k, v)]
        x = self.attend(q, k, v)
        x = self.view_out(x, n=self.n_heads, h=h, w=w)
        return self.to_out(x)


class ScaleAttend(nn.Module):
    """Scaled Dot-Product Attention
    attn(q, k, v) = softmax(qk'/sqrt(dk))*v"""

    def __init__(self, drop_prob=0., **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, q, k, v, attention_mask=None):
        scale = q.shape[-1] ** -0.5
        # similarity -> (..., i, j), usually i=j=s
        # sim = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scale
        sim = torch.matmul(q, k.transpose(-2, -1)) * scale
        sim = self.mask(sim, attention_mask)

        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)

        # attn = einsum('... i j, ... j d -> ... i d', attn, v)
        attn = torch.matmul(attn, v)

        return attn

    def mask(self, sim, attention_mask):
        if attention_mask is not None:  # mask pad
            # [[1,1,1,0,0,....]] -> [F,F,F,T,T,...]
            attention_mask = ~attention_mask.to(dtype=torch.bool)
            if len(attention_mask.shape) != len(sim.shape):
                *t, i, j = sim.shape
                n, j = attention_mask.shape
                attention_mask = attention_mask.view(n, *[1] * len(t), j).repeat(1, 1, i, 1)  # (n, j) -> (n, 1, i, j)
            sim = sim.masked_fill(attention_mask, torch.finfo(sim.dtype).min)  # support fp16
        return sim


class ScaleAttendWithXformers(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # faster and less memory
        # requires pytorch > 2.0
        from xformers.ops import memory_efficient_attention
        self.fn = memory_efficient_attention

    def forward(self, q, k, v, attention_mask=None, **kwargs):
        return self.fn(q, k, v, attention_mask=attention_mask, **kwargs)


class LinearAttend(nn.Module):
    """linear attention, to reduce the computation
    refer to: https://arxiv.org/pdf/2006.16236.pdf
    attn(q, k, v) = softmax(k') * v * softmax(q')/sqrt(dk)
    there, softmax(k') * v -> (d * d)
    where in original attention, softmax(qk') -> (s * s)
    when s >> d, got less computation
    """

    def __init__(self, **kwargs):
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
        from packaging import version
        assert version.parse(torch.__version__) >= version.parse("2.0.0")

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


class MemoryScaleAttend2D(ScaleAttend):
    def __init__(self, n_heads, n_mem_size, head_dim, drop_prob=0., **kwargs):
        super().__init__(drop_prob=drop_prob)
        self.mem_kv = nn.Parameter(torch.randn(2, n_heads, n_mem_size, head_dim))

    def forward(self, q, k, v, attention_mask=None):
        mk, mv = map(lambda t: repeat(t, 'n j d -> b n j d', b=q.shape[0]), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        return super().forward(q, k, v, attention_mask=attention_mask)


class MemoryScaleAttend3D(ScaleAttend):
    def __init__(self, n_heads, n_mem_size, head_dim, drop_prob=0., **kwargs):
        super().__init__(drop_prob=drop_prob)
        self.mem_kv = nn.Parameter(torch.randn(2, 1, n_mem_size, n_heads * head_dim))

    def forward(self, q, k, v, attention_mask=None):
        mk, mv = map(lambda t: repeat(t, 'n j d -> b n j d', b=q.shape[0]), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        return super().forward(q, k, v, attention_mask=attention_mask)


class MemoryLinearAttend(LinearAttend):
    def __init__(self, n_heads, head_dim, n_mem_size, drop_prob=0., **kwargs):
        super().__init__(drop_prob=drop_prob)
        self.mem_kv = nn.Parameter(torch.randn(2, n_heads, n_mem_size, head_dim))

    def forward(self, q, k, v, attention_mask=None):
        mk, mv = map(lambda t: repeat(t, 'n d j -> b n d j', b=q.shape[0]), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        return super().forward(q, k, v, attention_mask=attention_mask)
