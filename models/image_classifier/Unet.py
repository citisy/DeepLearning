import torch
import torch.nn as nn
from utils.layers import Conv, ConvInModule, ConvT

# in_ches, hidden_ches, out_ches
# bottom -> top
unet256_config = (
    [64 * 8, *[64 * 8] * 3, 64 * 4, 64 * 2, 64, 3],
    [64 * 8, *[64 * 8] * 3, 64 * 8, 64 * 4, 64 * 2, 64],
    [64 * 8, *[64 * 8] * 3, 64 * 4, 64 * 2, 64, 3],
)


class Model(nn.Module):
    def __init__(self, conv_config=unet256_config):
        super().__init__()

        in_ches, hidden_ches, out_ches = conv_config

        # bottom -> top
        submodule = CurBlock(in_ches, hidden_ches, out_ches)
        while not submodule.is_top_block:
            submodule = CurBlock(in_ches, hidden_ches, out_ches, submodule=submodule)

        self.conv_seq = submodule

    def forward(self, x):
        return self.conv_seq(x)


class CurBlock(nn.Module):
    def __init__(self, in_ches, hidden_ches, out_ches, submodule=None):
        super().__init__()
        in_ch, hidden_ch, out_ch = in_ches.pop(0), hidden_ches.pop(0), out_ches.pop(0)

        is_bottom_block = submodule is None
        is_top_block = len(in_ches) == 0

        layers = []

        # down
        if is_top_block:
            layers.append(Conv(in_ch, hidden_ch, k=4, s=2, p=1, is_act=False, is_norm=False, mode='acn'))
        elif is_bottom_block:
            layers.append(Conv(in_ch, hidden_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2, True), is_norm=False, mode='acn'))
        else:
            layers.append(Conv(in_ch, hidden_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2, True), mode='acn'))

        # sub
        if not is_bottom_block:
            layers.append(submodule)

        # up
        if is_top_block:
            layers.append(ConvT(hidden_ch * 2, out_ch, k=4, s=2, p=1, bias=True, is_norm=False, mode='acn'))
            layers.append(nn.Tanh())
        elif is_bottom_block:
            layers.append(ConvT(hidden_ch, out_ch, k=4, s=2, p=1, mode='acn'))
        else:
            layers.append(ConvT(hidden_ch * 2, out_ch, k=4, s=2, p=1, mode='acn'))

        self.conv_seq = nn.Sequential(*layers)
        self.is_bottom_block = is_bottom_block
        self.is_top_block = is_top_block

    def forward(self, x):
        y = self.conv_seq(x)
        if self.is_top_block:
            return y
        else:
            return torch.cat([x, y], 1)
