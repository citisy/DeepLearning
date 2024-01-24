import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from functools import partial
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce


class SimpleInModule(nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


class ConvInModule(nn.Sequential):
    def __init__(self, in_ch=3, input_size=224, out_ch=None, output_size=None):
        out_ch = out_ch or in_ch
        output_size = output_size or input_size

        assert in_ch <= out_ch, f'input channel must not be greater than {out_ch = }'
        assert input_size >= output_size, f'input size must not be smaller than {output_size = }'

        self.in_channels = in_ch
        self.out_channels = out_ch
        self.input_size = input_size

        if in_ch == out_ch and input_size == output_size:
            projection = nn.Identity()
        else:
            # in_ch -> out_ch
            # input_size -> output_size
            projection = nn.Conv2d(in_ch, out_ch, (input_size - output_size) + 1)

        super().__init__(projection)


class OutModule(nn.Sequential):
    def __init__(self, out_features, in_features=1000):
        assert out_features <= in_features, f'output features must not be greater than {in_features}'
        super().__init__(
            nn.Linear(in_features, out_features)
        )


class Conv(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s=1, p=None,
                 is_act=True, act=None,
                 is_norm=True, norm=None,
                 is_drop=True, drop_prob=0.5,
                 mode='cna', **conv_kwargs):
        """

        Args:
            in_ch (int): channel size
            out_ch (int): channel size
            k (int or tuple): kernel size
            s: stride
            p: padding size, None for full padding
            act (nn.Module): activation function
            mode (str): e.g. 'cna' gives conv - norm - act
                - 'c' gives convolution function
                - 'n' gives normalization function
                - 'a' gives activate function
                - 'd' gives dropout function

        """
        if p is None:
            p = self.auto_p(k, s) if isinstance(k, int) else [self.auto_p(x, s) for x in k]

        self.is_act = is_act
        self.is_norm = is_norm
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.kernel_size = k
        self.stride = s
        self.padding = p

        layers = OrderedDict()

        for i, m in enumerate(mode):
            if m == 'c':
                layers['conv'] = (nn.Conv2d(in_ch, out_ch, k, s, p, **conv_kwargs))
            elif m == 'n' and is_norm:
                if norm is None:
                    j = mode.index('c')
                    if i < j:  # norm first
                        norm_ch = in_ch
                    else:
                        norm_ch = out_ch
                    norm = nn.BatchNorm2d(norm_ch)
                layers['norm'] = norm
            elif m == 'a' and is_act:
                layers['act'] = act or nn.ReLU()
            elif m == 'd' and is_drop:
                layers['drop'] = nn.Dropout(drop_prob)

        super().__init__(layers)

    @staticmethod
    def auto_p(k, s):
        """auto pad to divisible totally
        o=i/s+(2p-k)/s+1 -> p=(k-s)/2
        e.g.
            input_size=224, k=3, s=2 if output_size=224/s=112, p=(k-s)/2=0.5 -> 1
        """
        return int(np.ceil((k - s) / 2)) if k > s else 0


class ConvT(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s=1, p=None,
                 is_act=True, act=None,
                 is_norm=True, norm=None,
                 is_drop=True, drop_prob=0.5,
                 mode='cna', only_upsample=False,
                 **conv_kwargs):
        """

        Args:
            in_ch (int): channel size
            out_ch (int): channel size
            k (int or tuple): kernel size
            s: stride
            p: padding size, None for full padding
            act (nn.Module): activation function
            mode (str):
                'c' gives convolution function, 'n' gives normalization function, 'a' gives activate function
                e.g. 'cna' gives conv - norm - act

        """
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.stride = s

        if only_upsample:
            # only upsample, no weight, can not be trained, but is smaller and quicker
            assert in_ch == out_ch, f'if {only_upsample = }, {in_ch = } must be equal to {out_ch = }'
            layers = [nn.Upsample(scale_factor=s, mode='bilinear', align_corners=False)]

        else:
            if p is None:
                p = self.auto_p(k, s) if isinstance(k, int) else [self.auto_p(x, s) for x in k]

            self.is_act = is_act
            self.is_norm = is_norm
            self.kernel_size = k
            self.padding = p

            layers = OrderedDict()

            for i, m in enumerate(mode):
                if m == 'c':
                    layers['conv'] = nn.ConvTranspose2d(in_ch, out_ch, k, s, p, **conv_kwargs)
                elif m == 'n' and is_norm:
                    if norm is None:
                        j = mode.index('c')
                        if i < j:  # norm first
                            norm_ch = in_ch
                        else:
                            norm_ch = out_ch
                        norm = nn.BatchNorm2d(norm_ch)
                    layers['norm'] = norm
                elif m == 'a' and is_act:
                    layers['act'] = act or nn.ReLU(True)
                elif m == 'd' and is_drop:
                    layers['drop'] = nn.Dropout(drop_prob)

        super().__init__(layers)

    @staticmethod
    def auto_p(k, s):
        """auto pad to divisible totally
        o=si+k-s-2p -> p=(k-s)/2
        e.g.
            input_size=224, k=4, s=2 if output_size=224*s=448, p=(k-s)/2=1
        """
        return int(np.ceil((k - s) / 2)) if k > s else 0


class Linear(nn.Sequential):
    def __init__(self, in_features, out_features,
                 linear=None,
                 is_act=True, act=None,
                 is_norm=True, norm=None,
                 is_drop=True, drop_prob=0.5,
                 mode='lna',
                 **linear_kwargs
                 ):
        self.is_act = is_act
        self.is_norm = is_norm
        self.is_drop = is_drop

        layers = OrderedDict()

        for i, m in enumerate(mode):
            if m == 'l':
                if linear is None:
                    linear = nn.Linear
                layers['linear'] = linear(in_features, out_features, **linear_kwargs)
            elif m == 'n' and is_norm:
                if norm is None:
                    j = mode.index('l')
                    if i < j:  # norm first
                        norm_features = in_features
                    else:
                        norm_features = out_features
                    norm = nn.BatchNorm1d(norm_features)
                layers['norm'] = norm
            elif m == 'a' and is_act:
                layers['act'] = act or nn.Sigmoid()
            elif m == 'd' and is_drop:
                layers['drop'] = nn.Dropout(drop_prob)

        self.in_features = in_features
        self.out_features = out_features
        super().__init__(layers)


class EqualLinear(nn.Module):
    def __init__(self, in_features, out_features, lr_mul=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


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


class SelfAttention2D(nn.Module):
    def __init__(self, n_heads=None, model_dim=None, head_dim=None, drop_prob=0.1):
        """self attention build by linear function
        attn(q, k, v) = softmax(qk'/sqrt(dk))*v

        Args:
            n_heads:
            model_dim: d_model
            head_dim: d_k
            drop_prob:
        """
        super().__init__()
        n_heads, model_dim, head_dim = get_attention_input(n_heads, model_dim, head_dim)
        self.scale = head_dim ** -0.5

        self.to_qkv = nn.ModuleList([nn.Sequential(
            nn.Linear(model_dim, model_dim),
            Rearrange('b s (n dk)-> b n s dk', dk=head_dim, n=n_heads)
        ) for _ in range(3)])
        self.dropout = nn.Dropout(drop_prob)
        self.to_out = nn.Sequential(
            Rearrange('b n s dk -> b s (n dk)'),
            Linear(model_dim, model_dim, mode='ld', drop_prob=drop_prob)
        )

    def attend(self, q, k, v, attention_mask=None):
        """Scaled Dot-Product Attention"""
        # similarity
        sim = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if attention_mask is not None:  # mask pad
            attention_mask = ~attention_mask.to(dtype=torch.bool)
            attention_mask = attention_mask[:, None, None].repeat(1, 1, sim.size(2), 1)  # mask pad
            sim = sim.masked_fill(attention_mask, torch.finfo(sim.dtype).min)  # support fp16

        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)
        attn = torch.matmul(attn, v)

        return attn

    def forward(self, x, attention_mask=None):
        q, k, v = [m(x) for m in self.to_qkv]
        x = self.attend(q, k, v, attention_mask=attention_mask)

        return self.to_out(x)


class SelfAttention3D(nn.Module):
    def __init__(self, in_dim, n_heads=None, model_dim=None, head_dim=None,
                 use_mem_kv=True, n_mem_size=4, drop_prob=0., **conv_kwargs):
        """self attention build by conv function
        attn(q, k, v) = softmax(qk'/sqrt(dk))*v

        Args:
            in_dim:
            n_heads:
            model_dim:
            head_dim:
            n_mem_size:
            drop_prob:
        """
        super().__init__()
        n_heads, model_dim, head_dim = get_attention_input(n_heads, model_dim, head_dim)
        self.scale = head_dim ** -0.5
        self.n_heads = n_heads
        self.use_mem_kv = use_mem_kv

        # different to linear function, each conv filter and feature map is independent
        # so can use a conv layer to compute, and then, chunk it
        self.to_qkv = nn.Conv2d(in_dim, model_dim * 3, 1, **conv_kwargs)

        if use_mem_kv:
            self.mem_kv = nn.Parameter(torch.randn(2, n_heads, n_mem_size, head_dim))

        self.dropout = nn.Dropout(drop_prob)
        self.to_out = nn.Conv2d(model_dim, in_dim, 1)

    def attend(self, q, k, v):
        """Scaled Dot-Product Attention"""
        # similarity
        # s means image size, s = i + j, i = h * w
        sim = torch.einsum(f"b n i d, b n s d -> b n i s", q, k) * self.scale

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # aggregate values
        attn = torch.einsum(f"b n i s, b n s d -> b n i d", attn, v)
        return attn

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (n d) h w -> b n (h w) d', n=self.n_heads), qkv)

        if self.use_mem_kv:
            mk, mv = map(lambda t: repeat(t, 'n j d -> b n j d', b=q.shape[0]), self.mem_kv)
            k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        x = self.attend(q, k, v)
        x = rearrange(x, 'b n (h w) d -> b (n d) h w', h=h, w=w)
        return self.to_out(x)


class LinearSelfAttention3D(nn.Module):
    def __init__(self, in_dim, n_heads=None, model_dim=None, head_dim=None,
                 use_mem_kv=True, n_mem_size=4, norm=None, **conv_kwargs):
        """linear self attention build by conv function, to reduce the computation
        attn(q, k, v) = softmax(k)*v*softmax(q)/sqrt(dk)
        refer to: https://arxiv.org/pdf/2006.16236.pdf

        Args:
            in_dim:
            n_heads:
            model_dim:
            head_dim:
            n_mem_size:
            norm:
        """
        super().__init__()
        n_heads, model_dim, head_dim = get_attention_input(n_heads, model_dim, head_dim)
        self.scale = head_dim ** -0.5
        self.n_heads = n_heads
        self.use_mem_kv = use_mem_kv

        self.to_qkv = nn.Conv2d(in_dim, model_dim * 3, 1, **conv_kwargs)

        if use_mem_kv:
            self.mem_kv = nn.Parameter(torch.randn(2, n_heads, head_dim, n_mem_size))

        self.to_out = Conv(model_dim, in_dim, 1, mode='cn', norm=norm)

    def attend(self, q, k, v):
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum('b n i s, b n j s -> b n i j', k, v)  # d = i = j
        context = torch.einsum('b n i j, b n i s -> b n j s', context, q)
        return context

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (n d) h w -> b n d (h w)', n=self.n_heads), qkv)

        if self.use_mem_kv:
            mk, mv = map(lambda t: repeat(t, 'n d j -> b n d j', b=b), self.mem_kv)
            k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        x = self.attend(q, k, v)
        x = rearrange(x, 'b n d (h w) -> b (n d) h w', n=self.n_heads, h=h, w=w)
        return self.to_out(x)


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


class Cache(nn.Module):
    def __init__(self, idx=None, replace=True):
        super().__init__()
        self.idx = idx
        self.replace = replace

    def forward(self, x, features: list):
        """f_i = x"""
        if self.idx is not None:
            if self.replace:
                features[self.idx] = x
            else:
                features.insert(self.idx, x)
        else:
            features.append(x)
        return x, features


class Concat(nn.Module):
    def __init__(self, idx, dim=1, replace=False):
        super().__init__()
        self.idx = idx
        self.dim = dim
        self.replace = replace

    def forward(self, x, features: list):
        """x <- concat(x, f_i)"""
        x = torch.cat([x, features[self.idx]], self.dim)

        if self.replace:
            features[self.idx] = x

        return x, features


class Add(nn.Module):
    def __init__(self, idx, replace=False):
        super().__init__()
        self.idx = idx
        self.replace = replace

    def forward(self, x, features: list):
        """x <- x + f_i"""
        x += features[self.idx]

        if self.replace:
            features[self.idx] = x

        return x, features


class Residual(nn.Module):
    def __init__(self, fn, project_fn=None, is_act=True, act=None):
        super().__init__()
        self.fn = fn
        self.project_fn = project_fn or nn.Identity()

        if is_act:
            self.act = act or nn.ReLU(True)
        else:
            self.act = nn.Identity()

    def forward(self, x, **kwargs):
        return self.act(self.fn(x, **kwargs) + self.project_fn(x))
