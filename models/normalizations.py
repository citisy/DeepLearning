import torch
from torch import nn

from utils import op_utils

make_norm_fn = op_utils.RegisterTables()
make_norm_fn.add_register()(nn.LayerNorm)
make_norm_fn.add_register()(nn.GroupNorm)


@make_norm_fn.add_register()
class RMSNorm(nn.Module):
    """
    dim:
        2 -> (b, s, d)  where channel_first=False
        3 -> (b, c, h, w) where channel_first=True
        4 -> (b, c, t, h, w) where channel_first=True
    """

    def __init__(self, num_channels, dim=2, bias=False, channel_first=False, eps=1e-6):
        super().__init__()
        if isinstance(num_channels, int):
            num_channels = (num_channels, *[1] * (dim - 1)) if channel_first else (num_channels,)
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels)) if bias else 0.
        self.eps = eps
        self.channel_first = channel_first

    def _norm(self, x):
        """norm(x) = x / \\sqrt{\\mean{x^2} + e}"""
        # same to
        # n = F.normalize(x, dim=1 if self.channel_first else -1, eps=self.eps) * (x.shape[1] ** 0.5)
        n = x * torch.rsqrt(x.pow(2).mean((1 if self.channel_first else -1), keepdim=True) + self.eps)
        return n

    def forward(self, x):
        n = self._norm(x.float())
        y = n * self.weight + self.bias
        return y.type_as(x)


@make_norm_fn.add_register()
class RMSNorm2D(RMSNorm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('dim', 2)
        kwargs.setdefault('channel_first', False)
        super().__init__(*args, **kwargs)


@make_norm_fn.add_register()
class RMSNorm3D(RMSNorm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('dim', 3)
        kwargs.setdefault('channel_first', True)
        super().__init__(*args, **kwargs)


@make_norm_fn.add_register()
class RMSNorm4D(RMSNorm):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('dim', 4)
        kwargs.setdefault('channel_first', True)
        super().__init__(*args, **kwargs)


@make_norm_fn.add_register()
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


@make_norm_fn.add_register()
class LayerNorm32(nn.LayerNorm):
    """forced to use fp32"""

    def forward(self, x):
        if self.elementwise_affine and self.weight.dtype is not torch.float32:
            # check the weight
            return super().forward(x)
        else:
            return super().forward(x.float()).type(x.dtype)


@make_norm_fn.add_register()
class GroupNorm32(nn.GroupNorm):
    """forced to use fp32"""

    def forward(self, x):
        if self.weight.dtype is not torch.float32:
            # check the weight
            return super().forward(x)
        else:
            return super().forward(x.float()).type(x.dtype)
