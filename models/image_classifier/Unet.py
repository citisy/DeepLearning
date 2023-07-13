import torch
import torch.nn as nn
from utils.layers import Conv, ConvInModule, ConvT

# top(outer) -> bottom(inner)
# in_ches, hidden_ches, out_ches
unet256_config = (
    [3, 64, 64 * 2, 64 * 4, *[64 * 8] * 3, 64 * 8],     # in_ches
    [64, 64*2, 64 * 4, 64 * 8, *[64 * 8] * 3, 64 * 8],  # hidden_ches
    [3, 64, 64 * 2, 64 * 4, *[64 * 8] * 3, 64 * 8],     # out_ches
)


class Model(nn.Module):
    def __init__(self, in_ch, input_size, in_module=None, out_module=None, conv_config=unet256_config):
        super().__init__()
        in_ches, hidden_ches, out_ches = conv_config

        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=in_ches[0])

        if out_module is None:
            out_module = nn.Sequential()

        self.input = in_module

        # top(outer) -> bottom(inner)
        self.conv_seq = CurBlock(in_ches, hidden_ches, out_ches, is_top_block=True)
        self.output = out_module

    def forward(self, x):
        x = self.input(x)
        x = self.conv_seq(x)
        x = self.output(x)
        return x


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
            layers.append(Conv(in_ch, hidden_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2, True), is_norm=False, mode='acn'))
        else:
            layers.append(Conv(in_ch, hidden_ch, k=4, s=2, p=1, act=nn.LeakyReLU(0.2, True), mode='acn'))

        # sub
        if not is_bottom_block:
            layers.append(CurBlock(in_ches, hidden_ches, out_ches))

        # up
        if is_top_block:
            layers.append(ConvT(hidden_ch * 2, out_ch, k=4, s=2, p=1, bias=True, is_norm=False, mode='acn'))
            layers.append(nn.Tanh())
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
