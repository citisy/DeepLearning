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


class LayerNorm2d(nn.Module):
    """
    # From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
    # Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
    """
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
