import math
import torch
from torch import nn


class FastGELU(nn.Module):
    """An approximation of gelu.

    See: https://arxiv.org/pdf/1606.08415.pdf
    """

    def forward(self, x):
        return 0.5 * x * (1. + torch.tanh(math.sqrt(2. / math.pi) * (x + 0.044715 * torch.pow(x, 3.))))
