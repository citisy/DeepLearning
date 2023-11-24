import torch
import torch.nn as nn
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
        self.backbone = CurBlock(in_ches, hidden_ches, out_ches)
        initialize_layers(self)

    def forward(self, x, pix_images=None):
        x = self.backbone(x)
        return super().forward(x, pix_images)


class CurBlock(nn.Module):
    def __init__(self, in_ches, hidden_ches, out_ches, layer_idx=0,
                 get_down_layer_func=None, get_up_layer_func=None):
        super().__init__()
        get_down_layer_func = get_down_layer_func or down_layers
        get_up_layer_func = get_up_layer_func or up_layers

        is_bottom = layer_idx == len(in_ches) - 1
        is_top = layer_idx == 0
        layers = []
        in_ch, hidden_ch, out_ch = in_ches[layer_idx], hidden_ches[layer_idx], out_ches[layer_idx]

        # down
        layers.append(get_down_layer_func(in_ch, hidden_ch, is_top, is_bottom, layer_idx))

        # sub
        if not is_bottom:
            layers.append(CurBlock(
                in_ches, hidden_ches, out_ches,
                layer_idx=layer_idx + 1,
                get_down_layer_func=get_down_layer_func,
                get_up_layer_func=get_up_layer_func
            ))

        # up
        layers.append(get_up_layer_func(hidden_ch, out_ch, is_top, is_bottom, layer_idx))

        self.conv_seq = nn.Sequential(*layers)
        self.is_bottom = is_bottom
        self.is_top = is_top

    def forward(self, x):
        y = self.conv_seq(x)
        if self.is_top:
            return y
        else:
            return torch.cat([x, y], 1)


def down_layers(in_ch, hidden_ch, is_top_block, is_bottom_block, layer_idx=None):
    if is_top_block:
        layer = Conv(in_ch, hidden_ch, k=4, s=2, p=1, mode='c')
    elif is_bottom_block:
        layer = Conv(in_ch, hidden_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2), mode='ac')
    else:
        layer = Conv(in_ch, hidden_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2), mode='acn')

    return layer


def up_layers(hidden_ch, out_ch, is_top_block, is_bottom_block, layer_idx=None):
    if is_top_block:
        layer = ConvT(hidden_ch * 2, out_ch, k=4, s=2, p=1, mode='ac')
    elif is_bottom_block:
        layer = ConvT(hidden_ch, out_ch, k=4, s=2, p=1, mode='acn')
    else:
        layer = ConvT(hidden_ch * 2, out_ch, k=4, s=2, p=1, mode='acn')

    return layer
