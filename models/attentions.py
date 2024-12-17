import math
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from utils import log_utils
from .layers import Linear


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


def get_mask(attention_mask, det_shape, det_dtype=None, return_bool=True):
    if attention_mask is None:
        return attention_mask

    # [1,1,1,0,0,....] -> [F,F,F,T,T,...]
    attention_mask = ~attention_mask.to(dtype=torch.bool)
    if len(attention_mask.shape) != len(det_shape):
        *t, i, j = det_shape
        n, j = attention_mask.shape
        attention_mask = attention_mask.view(n, *[1] * len(t), j).repeat(1, 1, i, 1)  # (n, j) -> (n, 1, i, j)

    if not return_bool:
        # [1,1,1,0,0,....] -> [0,0,0,-inf,-inf,,...]
        tmp = torch.zeros_like(attention_mask).to(attention_mask.device, dtype=det_dtype)
        tmp[attention_mask] = torch.finfo(det_dtype).min
        attention_mask = tmp

    return attention_mask


def mask(sim, attention_mask=None, use_min=True):
    if attention_mask is not None:  # mask pad
        attention_mask = get_mask(attention_mask, sim.shape)
        min_num = torch.finfo(sim.dtype).min if use_min else -float('inf')
        sim = sim.masked_fill(attention_mask, min_num)  # support fp16
    return sim


class CrossAttention2D(nn.Module):
    """cross attention"""

    def __init__(self, n_heads=None, model_dim=None, head_dim=None, query_dim=None, context_dim=None,
                 use_conv=False, separate=True, drop_prob=0.1, attend=None, out_layer=None, **fn_kwargs):
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
            self.to_out = nn.Conv1d(model_dim, query_dim, **fn_kwargs) if out_layer is None else out_layer

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
            self.to_out = Linear(model_dim, query_dim, mode='ld', drop_prob=drop_prob, **fn_kwargs) if out_layer is None else out_layer

        self.attend = ScaleAttend(drop_prob=drop_prob) if attend is None else attend

    def forward(self, q, k=None, v=None, attention_mask=None, **attend_kwargs):
        if self.separate:
            q, k, v = get_qkv(q, k, v)
            q, k, v = [m(x) for m, x in zip(self.to_qkv, (q, k, v))]
        else:
            dim = 1 if self.use_conv else -1
            q, k, v = self.to_qkv(q).chunk(3, dim=dim)

        q, k, v = [self.view_in(x).contiguous() for x in (q, k, v)]

        x = self.attend(q, k, v, attention_mask=attention_mask, **attend_kwargs)
        x = self.view_out(x)

        x = self.to_out(x)
        return x


class CrossAttention3D(nn.Module):
    """cross attention build by conv function"""

    def __init__(self, n_heads=None, model_dim=None, head_dim=None, query_dim=None, context_dim=None,
                 use_conv=True, separate=True, drop_prob=0., attend=None, out_layer=None, **fn_kwargs):
        super().__init__()
        n_heads, model_dim, head_dim = get_attention_input(n_heads, model_dim, head_dim)
        query_dim = query_dim or model_dim
        context_dim = context_dim or model_dim
        self.scale = head_dim ** -0.5
        self.n_heads = n_heads
        self.use_conv = use_conv
        self.separate = separate

        if use_conv:  # build by conv func
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
            self.view_out = partial(rearrange, pattern='b 1 (h w) c -> b c h w')
            self.to_out = nn.Conv2d(model_dim, query_dim, 1) if out_layer is None else out_layer

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

            self.view_in = Rearrange('b h w c -> b 1 (h w) c')

            self.view_out = partial(rearrange, pattern='b 1 (h w) c -> b h w c')
            self.to_out = Linear(model_dim, query_dim, mode='ld', drop_prob=drop_prob, **fn_kwargs) if out_layer is None else out_layer

        self.attend = ScaleAttend(drop_prob=drop_prob) if attend is None else attend

    def forward(self, q, k=None, v=None, **attend_kwargs):
        b, c, h, w = q.shape
        if self.separate:
            q, k, v = get_qkv(q, k, v)
            q, k, v = [m(x) for m, x in zip(self.to_qkv, (q, k, v))]
        else:  # only for self attention
            dim = 1 if self.use_conv else -1
            q, k, v = self.to_qkv(q).chunk(3, dim=dim)

        q, k, v = [self.view_in(x).contiguous() for x in (q, k, v)]

        x = self.attend(q, k, v, **attend_kwargs)
        x = self.view_out(x, h=h, w=w)
        return self.to_out(x)


class LinearAttention3D(nn.Module):
    """linear attention build by conv function"""

    def __init__(self, n_heads=None, model_dim=None, head_dim=None, query_dim=None, context_dim=None,
                 separate=True, attend=None, out_layer=None, **fn_kwargs):
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
        self.to_out = nn.Conv2d(model_dim, query_dim, 1) if out_layer is None else out_layer

    def forward(self, q, k=None, v=None, **attend_kwargs):
        b, c, h, w = q.shape
        if self.separate:
            q, k, v = get_qkv(q, k, v)
            q, k, v = [m(x) for m, x in zip(self.to_qkv, (q, k, v))]
        else:  # only for self attention
            q, k, v = self.to_qkv(q).chunk(3, dim=1)

        q, k, v = [self.view_in(x) for x in (q, k, v)]
        x = self.attend(q, k, v, **attend_kwargs)
        x = self.view_out(x, n=self.n_heads, h=h, w=w)
        return self.to_out(x)


class ScaleAttend(nn.Module):
    """Scaled Dot-Product Attention
    attn(q, k, v) = softmax(qk'/sqrt(dk))*v"""

    def __init__(self, drop_prob=0., **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, q, k, v, attention_mask=None, use_min=True, **kwargs):
        """
        in(q|k|v): (b n s d) or (b*n s d)
        out(attn): (b n s d) or (b*n s d)
        """
        scale = q.shape[-1] ** -0.5
        # similarity -> (..., i, j), usually i=j=s
        # sim = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scale
        sim = torch.matmul(q, k.transpose(-2, -1)) * scale
        sim = mask(sim, attention_mask, use_min=use_min)

        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)

        # attn = einsum('... i j, ... j d -> ... i d', attn, v)
        attn = torch.matmul(attn, v)

        return attn


class ScaleAttendWithXformers(nn.Module):
    def __init__(self, drop_prob=0., **kwargs):
        super().__init__()
        # faster and less memory
        # requires pytorch > 2.0
        from xformers.ops import memory_efficient_attention
        self.fn = partial(memory_efficient_attention, p=drop_prob)
        self.view_in = Rearrange('b n s d -> b s n d')
        self.view_out = Rearrange('b s n d -> b n s d')

    def forward(self, q, k, v, attention_mask=None, **kwargs):
        """
        in(q|k|v): (b n s d)
        out(attn): (b n s d)
        """
        q, k, v = [self.view_in(x) for x in (q, k, v)]
        attention_mask = get_mask(attention_mask, q.shape, q.dtype, return_bool=True)
        attn = self.fn(q, k, v, attn_bias=attention_mask, **kwargs)
        attn = self.view_out(attn)
        return attn


class SplitScaleAttend(ScaleAttend):
    """avoid out of memery
    todo: support mask"""

    def forward(self, q, k, v, **kwargs):
        assert not self.training, 'Do not support training mode yet'

        q_in_shape, k_in_shape, v_in_shape = q.shape, k.shape, v.shape
        q = q.view((-1, q_in_shape[-2], q_in_shape[-1]))
        k = k.view((-1, k_in_shape[-2], k_in_shape[-1]))
        v = v.view((-1, v_in_shape[-2], v_in_shape[-1]))

        scale = q.shape[-1] ** -0.5
        k = k * scale

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

        mem_free_total = log_utils.MemoryInfo.get_vram_info(q.device)['free_total']

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1

        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log2(mem_required / mem_free_total)))

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

        slice_size = q.shape[1] // steps
        for i in range(0, q.shape[1], slice_size):
            end = min(i + slice_size, q.shape[1])
            s1 = torch.einsum('b i d, b j d -> b i j', q[:, i:end], k)

            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1

            r1[:, i:end] = torch.einsum('b i j, b j d -> b i d', s2, v)
            del s2

        del q, k, v
        r1 = r1.view(q_in_shape)
        return r1


class LinearAttend(nn.Module):
    """linear attention, to reduce the computation
    refer to: https://arxiv.org/pdf/2006.16236.pdf
    attn(q, k, v) = softmax(k') * v * softmax(q')/sqrt(dk)
    there, softmax(k') * v -> (d * d)
    where in original attention, softmax(qk') -> (s * s)
    when s >> d, got less computation
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def attend(self, q, k, v, **kwargs):
        """
        in(q|k|v): (b n d s)
        out(attn): (b n d s)
        """
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        scale = q.shape[-2] ** -0.5
        q = q * scale

        context = torch.einsum('b n i s, b n j s -> b n i j', k, v)  # d = i = j
        context = torch.einsum('b n i j, b n i s -> b n j s', context, q)
        return context


class FlashAttend(nn.Module):
    """requires pytorch > 2.0
    refer to: https://arxiv.org/pdf/2205.14135.pdf

    note, use `pip install flash-attn --no-build-isolation` to install,
    will be very slow, suggest to use the follow step to install:
    ```
    git clone https://github.com/Dao-AILab/flash-attention
    cd flash-attention
    python setup.py install
    ```
    """

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

    def forward(self, q, k, v, **kwargs):
        """
        in(q|k|v): (b n s d)
        out(attn): (b n s d)
        """
        _, heads, q_len, _ = q.shape

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        # Check if there is a compatible device for flash attention
        config = self.cuda_config if q.is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale
        with torch.backends.cuda.sdp_kernel(**config):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.drop_prob if self.training else 0.
            )

        return out


class MemoryAttend(nn.Module):
    def __init__(self, base_layer=None, mem_kv=None, **kwargs):
        super().__init__()
        self.base_layer = base_layer
        self.mem_kv = mem_kv

    def forward(self, q, k, v, attention_mask=None, start_pos=0, **kwargs):
        """q,k,v: (b,n,s,d)"""
        k = self.cache(self.mem_kv[0], k, start_pos=start_pos)
        v = self.cache(self.mem_kv[1], v, start_pos=start_pos)
        return self.base_layer(q, k, v, attention_mask=attention_mask, **kwargs)

    @staticmethod
    def cache(mem_x, x, start_pos):
        b, _, s, _ = x.shape
        mem_x[:b, :, start_pos:start_pos + s, :] = x
        return mem_x[:b, :, :start_pos + s, :]


class MemoryScaleAttend2D(MemoryAttend):
    def __init__(self, n_heads, n_mem_size, head_dim, max_batch_size, **kwargs):
        mem_kv = torch.zeros(2, max_batch_size, n_heads, n_mem_size, head_dim)
        super().__init__(ScaleAttend(**kwargs), mem_kv)


class DynamicMemoryAttend(nn.Module):
    """don't alloc the kv memory when the class created
    only when the class called
    has dynamic cache len"""

    def __init__(self, base_layer=None, layer_idx=None, **kwargs):
        super().__init__()
        self.base_layer = base_layer

    def forward(self, q, k, v, attention_mask=None, cache_fn=None, **kwargs):
        """q,k,v: (b,n,s,d)"""
        k, v = cache_fn(k, v)  # only support inplace mode, see `cache_fn` to get more info
        return self.base_layer(q, k, v, attention_mask=attention_mask, **kwargs)

    @staticmethod
    def cache(k, v, past_kv={}, **kwargs):  # noqa
        """cache_fn example"""
        k = torch.cat([past_kv['k'], k], dim=-2)
        v = torch.cat([past_kv['v'], v], dim=-2)
        past_kv.update(
            k=k,
            v=v,
        )
        return k, v


class DynamicMemoryScaleAttend2D(DynamicMemoryAttend):
    def __init__(self, **kwargs):
        super().__init__(ScaleAttend(**kwargs))


class LearnedMemoryAttend(nn.Module):
    """apply learned kv cache"""

    def __init__(self, base_layer=None, mem_kv=None, is_repeat=True, **kwargs):
        super().__init__()
        self.base_layer = base_layer
        self.mem_kv = mem_kv
        self.is_repeat = is_repeat

    def get_mem_kv(self, k, v):
        b, *a = k.shape
        mem_kv = self.mem_kv
        if self.is_repeat:
            # (2, ...) -> (2, b, ...)
            mem_kv = mem_kv[:, None].repeat(1, b, *[1] * len(a))
        return mem_kv

    def forward(self, q, k, v, mem_kv=None, **kwargs):
        mem_kv = self.get_mem_kv(k, v) if mem_kv is None else mem_kv
        k, v = map(partial(torch.cat, dim=-2), ((mem_kv[0], k), (mem_kv[1], v)))

        return self.base_layer(q, k, v, **kwargs)


class LearnedMemoryScaleAttend2D(LearnedMemoryAttend):
    """apply learned kv cache for 2D scale attend
    in(q|k|v): (b n s d)
    out(attn): (b n s d)
    """

    def __init__(self, n_heads, n_mem_size, head_dim, **kwargs):
        mem_kv = nn.Parameter(torch.randn(2, n_heads, n_mem_size, head_dim))
        super().__init__(ScaleAttend(**kwargs), mem_kv)


class LearnedMemoryScaleAttend3D(LearnedMemoryAttend):
    """apply learned kv cache for 3D scale attend
    in(q|k|v): (b n s d)
    out(attn): (b n s d)
    """

    def __init__(self, n_heads, n_mem_size, head_dim, **kwargs):
        mem_kv = nn.Parameter(torch.randn(2, 1, n_mem_size, n_heads * head_dim))
        super().__init__(ScaleAttend(**kwargs), mem_kv)


class LearnedMemoryLinearAttend(LearnedMemoryAttend):
    """apply learned kv cache for linear attend
    in(q|k|v): (b n d s)
    out(attn): (b n d s)
    """

    def __init__(self, n_heads, n_mem_size, head_dim, **kwargs):
        mem_kv = nn.Parameter(torch.randn(2, n_heads, n_mem_size, head_dim))
        super().__init__(LinearAttend(**kwargs), mem_kv)


class RotaryAttend(ScaleAttend):
    def __init__(self, embedding=None, dim=None, **kwargs):
        super().__init__(**kwargs)
        if embedding is None:
            from .embeddings import RotaryEmbedding
            embedding = RotaryEmbedding(dim)

        self.view_in = Rearrange('b n s d -> b s n d')
        self.embedding = embedding
        self.view_out = Rearrange('b s n d -> b n s d')

    def forward(self, q, k, v, attention_mask=None, embedding_kwargs=dict(), **attend_kwargs):
        """
        in(q|k|v): (b n s d)
        out(attn): (b n s d)
        """
        q, k, v = [self.view_in(x).contiguous() for x in (q, k, v)]
        q = self.embedding(q, **embedding_kwargs)
        k = self.embedding(k, **embedding_kwargs)
        q, k, v = [self.view_out(x).contiguous() for x in (q, k, v)]
        attn = super().forward(q, k, v, attention_mask=attention_mask, **attend_kwargs)
        return attn


class MemoryRotaryAttend(ScaleAttend):
    def __init__(self, n_heads, n_mem_size, head_dim, max_batch_size, embedding=None, dim=None, **kwargs):
        super().__init__(**kwargs)
        if embedding is None:
            from .embeddings import RotaryEmbedding
            embedding = RotaryEmbedding(dim)

        self.view_in = Rearrange('b n s d -> b s n d')
        self.embedding = embedding
        self.view_out = Rearrange('b s n d -> b n s d')
        self.mem_kv = torch.zeros(2, max_batch_size, n_mem_size, n_heads, head_dim)

    def forward(self, q, k, v, start_pos=0, embedding_kwargs=dict(), **kwargs):
        """
        in(q|k|v): (b n s d)
        out(attn): (b n s d)
        """
        b, _, s, _ = q.shape
        q, k, v = [self.view_in(x).contiguous() for x in (q, k, v)]
        q = self.embedding(q, start_pos=start_pos, **embedding_kwargs)
        k = self.embedding(k, start_pos=start_pos, **embedding_kwargs)

        k = self.cache(self.mem_kv[0], k, start_pos=start_pos)
        v = self.cache(self.mem_kv[1], v, start_pos=start_pos)

        q, k, v = [self.view_out(x).contiguous() for x in (q, k, v)]
        attn = super().forward(q, k, v, **kwargs)
        return attn

    @staticmethod
    def cache(mem_x, x, start_pos):
        b, s, _, _ = x.shape
        mem_x[:b, start_pos:start_pos + s, :, :] = x
        return mem_x[:b, :start_pos + s, :, :]
