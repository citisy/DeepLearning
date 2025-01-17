import math
import torch
from torch import nn
from einops.layers.torch import Rearrange


class PositionalEmbedding(nn.Module):
    """
    emb_{2i} = sin{n * \theta^{-2d/D}}
    emb_{2i+1} = cos{n * \theta^{-2d/d}}
    where, n is token position of N, d is emb position of D
    """

    def __init__(self, num_embeddings, embedding_dim, theta=10000.):
        super().__init__()
        weight = torch.zeros(num_embeddings, embedding_dim).float()
        position = torch.arange(0, num_embeddings).float().unsqueeze(1)
        # exp{-d / D * log{\theta}} = \theta ^ {-d / D}
        # same to
        # div_term = 1.0 / (theta ** (torch.arange(0, embedding_dim, 2)[: (embedding_dim // 2)].float() / embedding_dim))
        div_term = (torch.arange(0, embedding_dim, 2).float() * -(math.log(theta) / embedding_dim)).exp()

        weight[:, 0::2] = torch.sin(position * div_term)
        weight[:, 1::2] = torch.cos(position * div_term)

        weight = weight.unsqueeze(0)
        self.register_buffer('weight', weight)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return self.weight[:, :x.shape[1]]


class LearnedPositionEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.register_buffer("position_ids", torch.arange(num_embeddings).expand((1, -1)), persistent=False)

    def forward(self, x):
        """make sure x.ndim >= 2 and x.shape[1] is seq_len"""
        return super().forward(self.position_ids[:, :x.shape[1]])


class SinusoidalEmbedding(nn.Module):
    """
    emb_{2i} = sin{x * \theta^{-2d/D}}
    emb_{2i+1} = cos{x * \theta^{-2d/D}}
    where, x is seq vec, d is emb position of D
    """

    def __init__(self, embedding_dim, theta=10000.):
        super().__init__()
        weights = torch.zeros(embedding_dim).float()
        # exp{-d / D * log{\theta}} = \theta ^ {-d / D}
        div_term = (torch.arange(0, embedding_dim, 2).float() * -(math.log(theta) / embedding_dim)).exp()

        weights[0::2] = torch.sin(div_term)
        weights[1::2] = torch.cos(div_term)

        self.register_buffer('weights', weights)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        emb = x[:, None] * self.weights[None, :]
        return emb


class LearnedSinusoidalEmbedding(nn.Module):
    """following @crowsonkb's lead with random (learned optional) sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim).unsqueeze(0), requires_grad=not is_random)

    def forward(self, x):
        x = x[:, None]
        freqs = x * self.weights * 2 * torch.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class RotaryEmbedding(nn.Module):
    """for qk of rotary attention, not for seq"""
    def __init__(self, embedding_dim, theta=10000.):
        super().__init__()

        # exp{-d / D * log{\theta}} = \theta ^ {-d / D} = 1 / (\theta ^ {d / D})
        # equal to
        # div_term = 1.0 / (theta ** (torch.arange(0, embedding_dim, 2)[: (embedding_dim // 2)].float() / embedding_dim))
        # but will
        div_term = (torch.arange(0, embedding_dim, 2).float() * -(math.log(theta) / embedding_dim)).exp()
        self.div_term = div_term

    def make_weights(self, seq_len):
        position = torch.arange(0, seq_len).float()
        freqs = torch.outer(position, self.div_term).float()

        weights = torch.polar(torch.ones_like(freqs), freqs)
        return weights

    def forward(self, x, start_pos=0, weights=None):
        """x: (b s n d)
        y_{d-1} = x_{d-1}cos(w_{d/2}) - x_{d}sin(w_{d/2}), d in {1,3,5,...}
        y_{d} = x_{d}cos(w_{d/2}) + x_{d-1}sin(w_{d/2}), d in {2,4,6,...}
        """
        if weights is None:
            weights = self.make_weights(x.shape[1])

        weights = weights.to(x.device)
        s = x.shape[1]
        weights = weights[None, :, None, :]  # (s d) -> (1 s 1 d)
        weights = weights[:, start_pos:start_pos + s, :, :]

        # equal to:
        # weights = weights.repeat_interleave(2, -1)
        # weights = torch.view_as_real(weights)
        # cos = weights[..., 0]
        # sin = weights[..., 1]
        # x = x.float()
        # _x = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2, -1)
        # y = (x * cos) + (_x * sin)
        y = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        y = torch.view_as_real(y * weights).flatten(3)
        return y.type_as(x)


class PatchEmbedding(nn.Module):
    """embedding for image"""

    def __init__(self, dim, patch_size, in_ch=3, bias=False, out_ndim=3):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Conv2d(in_ch, dim, patch_size, stride=patch_size, bias=bias),
            Rearrange('b c h w -> b (h w) c') if out_ndim == 3 else Rearrange('b c h w -> b h w c')
        )

    def forward(self, x):
        return self.fn(x)
