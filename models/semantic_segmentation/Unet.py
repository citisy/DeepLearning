import torch
import torch.nn as nn
from torch.nn import functional as F
from . import BaseSemSegModel
from ..layers import Conv, ConvInModule, ConvT
from utils.torch_utils import initialize_layers

# top(outer) -> bottom(inner)
# in_ches, hidden_ches, out_ches
unet256_config = (
    [3, 64, 64 * 2, 64 * 4, *[64 * 8] * 3, 64 * 8],  # in_ches
    [64, 64 * 2, 64 * 4, 64 * 8, *[64 * 8] * 3, 64 * 8],  # hidden_ches
    [3, 64, 64 * 2, 64 * 4, *[64 * 8] * 3, 64 * 8],  # out_ches
)


class Model(BaseSemSegModel):
    """refer to [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf)"""

    def __init__(self, in_ch, out_features, conv_config=unet256_config, **kwargs):
        super().__init__()
        in_ches, hidden_ches, out_ches = conv_config

        self.out_features = out_features + 1
        in_ches[0] = in_ch
        out_ches[0] = self.out_features
        # top(outer) -> bottom(inner)
        self.backbone = CurBlock(in_ches, hidden_ches, out_ches, is_top_block=True)
        initialize_layers(self)

    def forward(self, x, pix_images=None):
        x = self.backbone(x)
        return super().forward(x, pix_images)


class PureModel(nn.Sequential):
    def __init__(self, conv_config=unet256_config):
        in_ches, hidden_ches, out_ches = conv_config
        super().__init__(
            CurBlock(in_ches, hidden_ches, out_ches, is_top_block=True),
            nn.Tanh()
        )
        initialize_layers(self)


class CurBlock(nn.Module):
    def __init__(self, in_ches, hidden_ches, out_ches, is_top_block=False):
        super().__init__()
        in_ch, hidden_ch, out_ch = in_ches.pop(0), hidden_ches.pop(0), out_ches.pop(0)

        is_bottom_block = len(in_ches) == 0
        self.is_bottom_block = is_bottom_block
        self.is_top_block = is_top_block

        layers = []

        # down
        if is_top_block:
            layers.append(Conv(in_ch, hidden_ch, k=4, s=2, p=1, is_act=False, is_norm=False, mode='acn'))
        elif is_bottom_block:
            layers.append(Conv(in_ch, hidden_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2), is_norm=False, mode='acn'))
        else:
            layers.append(Conv(in_ch, hidden_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2), mode='acn'))

        # sub
        if not is_bottom_block:
            layers.append(CurBlock(in_ches, hidden_ches, out_ches))

        # up
        if is_top_block:
            layers.append(ConvT(hidden_ch * 2, out_ch, k=4, s=2, p=1, is_norm=False, mode='acn'))
        elif is_bottom_block:
            layers.append(ConvT(hidden_ch, out_ch, k=4, s=2, p=1, mode='acn'))
        else:
            layers.append(ConvT(hidden_ch * 2, out_ch, k=4, s=2, p=1, mode='acn'))

        self.conv_seq = nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv_seq(x)
        if self.is_top_block:
            return y
        else:
            return torch.cat([x, y], 1)
