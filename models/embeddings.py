import math
import torch
from torch import nn
from einops.layers.torch import Rearrange


class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        pe = torch.zeros(num_embeddings, embedding_dim).float()
        position = torch.arange(0, num_embeddings).float().unsqueeze(1)
        div_term = (torch.arange(0, embedding_dim, 2).float() * -(math.log(10000.0) / embedding_dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class LearnedPositionEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.register_buffer("position_ids", torch.arange(num_embeddings).expand((1, -1)))

    def forward(self, x):
        """make sure x.ndim >= 2 and x.shape[1] is seq_len"""
        return super().forward(self.position_ids[:, :x.shape[1]])


class SinusoidalPositionEmbedding(nn.Module):
    """
    emb_2i = sin{x * \theta^{-2i/d}}
    emb_{2i+1} = cos{x * \theta^{-2i/d}}
    """

    def __init__(self, dim, theta=10000):
        super().__init__()
        half_dim = dim // 2
        log_theta = math.log(theta)
        emb = log_theta / half_dim
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer('emb', emb)

    def forward(self, x):
        emb = self.emb
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.cos(), emb.sin()), dim=-1)
        return emb


class LearnedSinusoidalPositionEmbedding(nn.Module):
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
