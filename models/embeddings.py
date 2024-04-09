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
        self.register_buffer("position_ids", torch.arange(num_embeddings).expand((1, -1)))

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
        freqs = x[:, None] * self.weights[None, :]
        emb = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
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
    def __init__(self, num_embeddings, embedding_dim, theta=10000.):
        super().__init__()
        position = torch.arange(0, num_embeddings).float()

        # exp{-d / D * log{\theta}} = \theta ^ {-d / D}
        # same to
        # div_term = 1.0 / (theta ** (torch.arange(0, embedding_dim, 2)[: (embedding_dim // 2)].float() / embedding_dim))
        div_term = (torch.arange(0, embedding_dim, 2).float() * -(math.log(theta) / embedding_dim)).exp()
        freqs = torch.outer(position, div_term).float()  # type: ignore
        weights = torch.polar(torch.ones_like(freqs), freqs)
        weights = weights[None, :, None, :]  # (s d) -> (1 s 1 d)
        self.register_buffer('weights', weights)

    def forward(self, x, start_pos=0):
        """x: (b s n d)"""
        s = x.shape[1]
        weights = self.weights[:, start_pos:start_pos + s, :, :]
        y = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        y = torch.view_as_real(y * weights).flatten(3)
        return y.type_as(x)


class PatchEmbedding(nn.Module):
    """embedding for image"""

    def __init__(self, dim, patch_size):
        super().__init__()
        self.fn = nn.Sequential(
            nn.Conv2d(3, dim, patch_size, stride=patch_size, bias=False),
            Rearrange('b c h w -> b (h w) c')
        )

    def forward(self, x):
        return self.fn(x)
