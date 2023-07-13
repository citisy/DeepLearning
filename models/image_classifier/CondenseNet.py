import torch
from torch import nn
from utils.layers import Conv, Linear, ConvInModule, OutModule
from .DenseNet import DenseBlock, Transition, DenseNet
from .ShuffleNet import shuffle, GConv

# (n_conv, growth_rate)
config = ((4, 8), (6, 16), (8, 32), (10, 64), (8, 128))

progress = 0


class CondenseNet(DenseNet):
    """[CondenseNet: An Efficient DenseNet using Learned Group Convolutions](https://arxiv.org/pdf/1711.09224.pdf)"""

    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            groups=4, conv_config=config, lgc_config=dict()
    ):
        super().__init__(in_ch, input_size, output_size, in_module, out_module, conv_config=conv_config)

        for i, module in enumerate(self.conv_seq):
            if isinstance(module, DenseBlock):
                self.conv_seq[i] = ConDenseBlock(
                    module.in_channels, module.growth_rate, module.n_conv,
                    g=groups, lgc_config=lgc_config
                )


class ConDenseBlock(nn.Module):
    def __init__(
            self, in_ch, growth_rate,
            n_conv=2, g=1, lgc_config=dict()
    ):
        super().__init__()

        layers = []

        for _ in range(n_conv):
            layers.append(nn.Sequential(
                LGConv(in_ch, 4 * growth_rate, k=1, g=g, **lgc_config),
                GConv(4 * growth_rate, growth_rate, k=3, g=g)
            ))
            in_ch += growth_rate

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for conv in self.layers:
            xl = conv(x)
            x = torch.cat((x, xl), dim=1)

        return x


class LGConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=None, g=1, act=nn.ReLU,
                 condense_factor=4, drop_prob=0.5, max_epochs=100,
                 is_norm=True, conv_kwargs=dict(),
                 ):
        super().__init__()
        self.condense_factor = condense_factor
        self.g = g
        self.stage = 1  # max_stage = condense_factor
        self.max_epochs = max_epochs

        self.conv = Conv(
            in_ch, out_ch, k, s=s, p=p, act=act,
            is_norm=is_norm,
            conv_kwargs=conv_kwargs
        )
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(True)
        self.drop = nn.Dropout(drop_prob)

        self.register_buffer('mask', torch.ones(self.conv.conv.weight.size()))
        self.c = 0
        self.filter_num = in_ch // condense_factor

    def forward(self, x):
        # progress = epoch / max_epoch
        # max_progress = 2 / (self.condense_factor - 1)
        stage = min(progress * 2 * (self.condense_factor - 1), self.condense_factor)

        if stage > self.stage:
            self.stage = stage
            self.mask_weight()

        self.conv.conv.weight.data *= self.mask
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)

        return x

    def mask_weight(self):
        weight = self.conv.conv.weight.data
        weight *= self.mask
        weight = weight.abs()
        weight = shuffle_weight(weight, self.g)

        o, i, _, _ = weight.size()
        c_ = o // self.g
        for g in range(self.g):
            wi = weight[g * c_:(g + 1) * c_]

            # l1 norm
            di = wi.sum((0, 2, 3))

            # filter filter_num channels
            # [:self.c] has been set to 0
            di = di.sort()[1][self.c:self.c + self.filter_num]

            for d in di.data:
                self.mask[i::self.g, d, :, :].fill_(0)

        self.c += self.filter_num

    @property
    def lasso_loss(self):
        if self.stage >= self.g - 1:
            return 0

        weight = self.conv.conv.weight.data * self.mask

        o, i, k1, k2 = weight.size()

        weight = weight.pow(2)
        c_ = o // self.g

        weight = weight.view(c_, self.g, i, k1, k2)
        weight = weight.sum((0, 2, 3, 4)).clamp(min=1e-6).sqrt()
        loss = weight.sum()
        return loss


def shuffle_weight(weight, g):
    o, i, k1, k2 = weight.size()  # out_ch, in_ch, k, k
    c_ = o // g
    weight = weight.view(c_, g, i, k1, k2)
    weight = weight.transpose(0, 1).contiguous()
    weight = weight.view(o, i, k1, k2)

    return weight


Model = CondenseNet
