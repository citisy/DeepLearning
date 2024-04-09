import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm2D(nn.Module):
    """input: (b, s, d)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x):
        # n = F.normalize(x, dim=2, eps=self.eps) * (x.shape[1] ** 0.5)
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return n

    def forward(self, x):
        n = self._norm(x.float()).type_as(x)
        return n * self.weight


class RMSNorm3D(nn.Module):
    """input: (b, c, h, w)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.eps = eps

    def _norm(self, x):
        # n = F.normalize(x, dim=1, eps=self.eps) * (x.shape[1] ** 0.5)
        n = x * torch.rsqrt(x.pow(1).mean(1, keepdim=True) + self.eps)
        return n

    def forward(self, x):
        n = self._norm(x.float()).type_as(x)
        return n * self.weight
