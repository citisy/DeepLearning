import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class SimpleInModule(nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


class ConvInModule(nn.Sequential):
    def __init__(self, in_ch=3, input_size=224, out_ch=None, output_size=None):
        out_ch = out_ch or in_ch
        output_size = output_size or input_size

        assert in_ch <= out_ch, f'input channel must not be greater than {out_ch = }'
        assert input_size >= output_size, f'input size must not be smaller than {output_size = }'

        self.in_channels = in_ch
        self.out_channels = out_ch
        self.input_size = input_size

        if in_ch == out_ch and input_size == output_size:
            projection = nn.Identity()
        else:
            # in_ch -> out_ch
            # input_size -> output_size
            projection = nn.Conv2d(in_ch, out_ch, (input_size - output_size) + 1)

        super().__init__(projection)


class OutModule(nn.Sequential):
    def __init__(self, out_features, in_features=1000):
        assert out_features <= in_features, f'output features must not be greater than {in_features}'
        super().__init__(
            nn.Linear(in_features, out_features)
        )


class Conv(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s=1, p=None, bias=False,
                 is_act=True, act=None,
                 is_norm=True, norm=None,
                 is_drop=True, drop_prob=0.7,
                 mode='cna', **conv_kwargs):
        """

        Args:
            in_ch (int): channel size
            out_ch (int): channel size
            k (int or tuple): kernel size
            s: stride
            p: padding size, None for full padding
            act (nn.Module): activation function
            mode (str):
                'c' gives convolution function, 'n' gives normalization function, 'a' gives activate function
                e.g. 'cna' gives conv - norm - act

        """
        if p is None:
            p = self.auto_p(k, s) if isinstance(k, int) else [self.auto_p(x, s) for x in k]

        self.is_act = is_act
        self.is_norm = is_norm
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.kernel_size = k
        self.stride = s
        self.padding = p

        layers = []

        for m in mode:
            if m == 'c':
                layers.append(nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias, **conv_kwargs))
            elif m == 'n' and is_norm:
                layers.append(norm or nn.BatchNorm2d(out_ch))
            elif m == 'a' and is_act:
                layers.append(act or nn.ReLU(True))
            elif m == 'd' and is_drop:
                layers.append(nn.Dropout(drop_prob))

        super().__init__(*layers)

    @staticmethod
    def auto_p(k, s):
        """auto pad to divisible totally
        o=i/s+(2p-k)/s+1 -> p=(k-s)/2
        e.g.
            input_size=224, k=3, s=2 if output_size=224/s=112, p=(k-s)/2=0.5 -> 1
        """
        return int(np.ceil((k - s) / 2)) if k > s else 0


class ConvT(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s=1, p=None, bias=False,
                 is_act=True, act=None,
                 is_norm=True, norm=None,
                 is_drop=True, drop_prob=0.7,
                 mode='cna', only_upsample=False,
                 **conv_kwargs):
        """

        Args:
            in_ch (int): channel size
            out_ch (int): channel size
            k (int or tuple): kernel size
            s: stride
            p: padding size, None for full padding
            act (nn.Module): activation function
            mode (str):
                'c' gives convolution function, 'n' gives normalization function, 'a' gives activate function
                e.g. 'cna' gives conv - norm - act

        """
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.stride = s

        if only_upsample:
            # only upsample, no weight, can not be trained, but is smaller and quicker
            assert in_ch == out_ch, f'if {only_upsample = }, {in_ch = } must be equal to {out_ch = }'
            layers = [nn.Upsample(scale_factor=s, mode='bilinear', align_corners=False)]

        else:
            if p is None:
                p = self.auto_p(k, s) if isinstance(k, int) else [self.auto_p(x, s) for x in k]

            self.is_act = is_act
            self.is_norm = is_norm
            self.kernel_size = k
            self.padding = p

            layers = []

            for m in mode:
                if m == 'c':
                    layers.append(nn.ConvTranspose2d(in_ch, out_ch, k, s, p, bias=bias, **conv_kwargs))
                elif m == 'n' and is_norm:
                    layers.append(norm or nn.BatchNorm2d(out_ch))
                elif m == 'a' and is_act:
                    layers.append(act or nn.ReLU(True))
                elif m == 'd' and is_drop:
                    layers.append(nn.Dropout(drop_prob))

        super().__init__(*layers)

    @staticmethod
    def auto_p(k, s):
        """auto pad to divisible totally
        o=si+k-s-2p -> p=(k-s)/2
        e.g.
            input_size=224, k=4, s=2 if output_size=224*s=448, p=(k-s)/2=1
        """
        return int(np.ceil((k - s) / 2)) if k > s else 0


class Linear(nn.Sequential):
    def __init__(self, in_features, out_features,
                 linear=None,
                 is_act=True, act=None,
                 is_norm=True, norm=None,
                 is_drop=True, drop_prob=0.7,
                 mode='lna',
                 **linear_kwargs
                 ):
        self.is_act = is_act
        self.is_norm = is_norm
        self.is_drop = is_drop

        layers = []

        for m in mode:
            if m == 'l':
                if linear is None:
                    linear = nn.Linear
                layers.append(linear(in_features, out_features, **linear_kwargs))
            elif m == 'n' and is_norm:
                layers.append(norm or nn.BatchNorm1d(out_features))
            elif m == 'a' and is_act:
                layers.append(act or nn.Sigmoid())
            elif m == 'd' and is_drop:
                layers.append(nn.Dropout(drop_prob))

        self.in_features = in_features
        self.out_features = out_features
        super().__init__(*layers)


class EqualLinear(nn.Module):
    def __init__(self, in_features, out_features, lr_mul=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))

        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class Cache(nn.Module):
    def __init__(self, idx=None, replace=True):
        super().__init__()
        self.idx = idx
        self.replace = replace

    def forward(self, x, features: list):
        if self.idx is not None:
            if self.replace:
                features[self.idx] = x
            else:
                features.insert(self.idx, x)
        else:
            features.append(x)
        return x, features


class Concat(nn.Module):
    def __init__(self, idx, dim=1, replace=False):
        super().__init__()
        self.idx = idx
        self.dim = dim
        self.replace = replace

    def forward(self, x, features):
        x = torch.cat([x, features[self.idx]], self.dim)

        if self.replace:
            features[self.idx] = x

        return x, features


class Add(nn.Module):
    def __init__(self, idx, replace=False):
        super().__init__()
        self.idx = idx
        self.replace = replace

    def forward(self, x, features):
        x += features[self.idx]

        if self.replace:
            features[self.idx] = x

        return x, features


class Residual(nn.Module):
    def __init__(self, fn, project_fn=None,
                 is_act=True, act=None):
        super().__init__()
        self.fn = fn
        self.project_fn = project_fn or nn.Identity()

        if is_act:
            self.act = act or nn.ReLU(True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.fn(x) + self.project_fn(x))
