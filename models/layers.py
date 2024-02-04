import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


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
    def __init__(self, in_ch, out_ch, k, s=1, p=None,
                 is_act=True, act=None,
                 is_norm=True, norm=None,
                 is_drop=True, drop_prob=0.5,
                 mode='cna', detail_name=True, **conv_kwargs):
        """

        Args:
            in_ch (int): channel size
            out_ch (int): channel size
            k (int or tuple): kernel size
            s: stride
            p: padding size, None for full padding
            act (nn.Module): activation function
            mode (str): e.g. 'cna' gives conv - norm - act
                - 'c' gives convolution function
                - 'n' gives normalization function
                - 'a' gives activate function
                - 'd' gives dropout function

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

        layers = OrderedDict()

        for i, m in enumerate(mode):
            if m == 'c':
                layers['conv'] = (nn.Conv2d(in_ch, out_ch, k, s, p, **conv_kwargs))
            elif m == 'n' and is_norm:
                if norm is None:
                    j = mode.index('c')
                    if i < j:  # norm first
                        norm_ch = in_ch
                    else:
                        norm_ch = out_ch
                    norm = nn.BatchNorm2d(norm_ch)
                layers['norm'] = norm
            elif m == 'a' and is_act:
                layers['act'] = act or nn.ReLU()
            elif m == 'd' and is_drop:
                layers['drop'] = nn.Dropout(drop_prob)

        if detail_name:
            super().__init__(layers)
        else:
            super().__init__(*[v for v in layers.values()])

    @staticmethod
    def auto_p(k, s):
        """auto pad to divisible totally
        o=i/s+(2p-k)/s+1 -> p=(k-s)/2
        e.g.
            input_size=224, k=3, s=2 if output_size=224/s=112, p=(k-s)/2=0.5 -> 1
        """
        return int(np.ceil((k - s) / 2)) if k > s else 0


class ConvT(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s=1, p=None,
                 is_act=True, act=None,
                 is_norm=True, norm=None,
                 is_drop=True, drop_prob=0.5,
                 mode='cna', only_upsample=False,
                 detail_name=True,
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

            layers = OrderedDict()

            for i, m in enumerate(mode):
                if m == 'c':
                    layers['conv'] = nn.ConvTranspose2d(in_ch, out_ch, k, s, p, **conv_kwargs)
                elif m == 'n' and is_norm:
                    if norm is None:
                        j = mode.index('c')
                        if i < j:  # norm first
                            norm_ch = in_ch
                        else:
                            norm_ch = out_ch
                        norm = nn.BatchNorm2d(norm_ch)
                    layers['norm'] = norm
                elif m == 'a' and is_act:
                    layers['act'] = act or nn.ReLU(True)
                elif m == 'd' and is_drop:
                    layers['drop'] = nn.Dropout(drop_prob)

        if detail_name:
            super().__init__(layers)
        else:
            super().__init__(*[v for v in layers.values()])

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
                 is_drop=True, drop_prob=0.5,
                 mode='lna', detail_name=True,
                 **linear_kwargs
                 ):
        self.is_act = is_act
        self.is_norm = is_norm
        self.is_drop = is_drop

        layers = OrderedDict()

        for i, m in enumerate(mode):
            if m == 'l':
                if linear is None:
                    linear = nn.Linear
                layers['linear'] = linear(in_features, out_features, **linear_kwargs)
            elif m == 'n' and is_norm:
                if norm is None:
                    j = mode.index('l')
                    if i < j:  # norm first
                        norm_features = in_features
                    else:
                        norm_features = out_features
                    norm = nn.BatchNorm1d(norm_features)
                layers['norm'] = norm
            elif m == 'a' and is_act:
                layers['act'] = act or nn.Sigmoid()
            elif m == 'd' and is_drop:
                layers['drop'] = nn.Dropout(drop_prob)

        self.in_features = in_features
        self.out_features = out_features
        if detail_name:
            super().__init__(layers)
        else:
            super().__init__(*[v for v in layers.values()])


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


class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch=None, use_conv=True):
        super().__init__()
        out_ch = out_ch or in_ch
        if use_conv:
            self.op = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        else:
            assert in_ch == out_ch
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch=None, use_conv=True):
        super().__init__()
        out_ch = out_ch or in_ch

        if use_conv:
            self.op = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, 3, padding=1)
            )
        else:
            assert in_ch == out_ch
            self.op = nn.Upsample(scale_factor=2)

        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x):
        return self.op(x)


class Cache(nn.Module):
    def __init__(self, idx=None, replace=True):
        super().__init__()
        self.idx = idx
        self.replace = replace

    def forward(self, x, features: list):
        """f_i = x"""
        if self.idx is not None:
            if self.replace:
                features[self.idx] = x
            else:
                features.insert(self.idx, x)
        else:
            features.append(x)
        return x, features


class Concat(nn.Module):
    def __init__(self, idx=-1, dim=1, replace=False, pop=False):
        super().__init__()
        self.idx = idx
        self.dim = dim
        self.replace = replace
        self.pop = pop

    def forward(self, x, features: list):
        """x <- concat(x, f_i)"""
        x = torch.cat([x, features[self.idx]], self.dim)

        if self.replace:
            features[self.idx] = x

        if self.pop:
            features.pop(self.idx)

        return x, features


class Add(nn.Module):
    def __init__(self, idx=-1, replace=False, pop=False):
        super().__init__()
        self.idx = idx
        self.replace = replace
        self.pop = pop

    def forward(self, x, features: list):
        """x <- x + f_i"""
        x += features[self.idx]

        if self.replace:
            features[self.idx] = x

        if self.pop:
            features.pop(self.idx)

        return x, features


class Residual(nn.Module):
    def __init__(self, fn, project_fn=None, is_act=True, act=None):
        super().__init__()
        self.fn = fn
        self.project_fn = project_fn or nn.Identity()

        if is_act:
            self.act = act or nn.ReLU(True)
        else:
            self.act = nn.Identity()

    def forward(self, x, **kwargs):
        return self.act(self.fn(x, **kwargs) + self.project_fn(x))
