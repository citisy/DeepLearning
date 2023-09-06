import torch
from torch import nn
import torch.nn.functional as F
from utils.torch_utils import initialize_layers
import numpy as np


class BaseImgClsModel(nn.Module):
    """a template to make a image classifier model by yourself"""

    def __init__(
            self,
            in_ch=3, input_size=None, out_features=None,
            in_module=None, backbone=None, neck=None, head=None, out_module=None,
            backbone_in_ch=3, backbone_input_size=224, head_hidden_features=1000, drop_prob=0.4
    ):
        super().__init__()

        self.in_channels = in_ch
        self.input_size = input_size
        self.out_features = out_features

        # `bool(nn.Sequential()) = False`, so do not ues `input = in_module or ConvInModule()`
        self.input = in_module if in_module is not None else ConvInModule(in_ch, input_size, out_ch=backbone_in_ch, output_size=backbone_input_size)
        self.backbone = backbone
        self.neck = neck if neck is not None else nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.head = head if head is not None else nn.Sequential(
            Linear(self.backbone.out_channels, head_hidden_features, is_drop=True, drop_prob=drop_prob),
            out_module or OutModule(out_features, in_features=head_hidden_features)
        )

        initialize_layers(self)

    def forward(self, x, true_label=None):
        x = self.input(x)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        loss = None
        if self.training:
            loss = self.loss(pred_label=x, true_label=true_label)

        return {'pred': x, 'loss': loss}

    def loss(self, pred_label, true_label):
        return F.cross_entropy(pred_label, true_label)


class BaseObjectDetectionModel(nn.Sequential):
    """a template to make a object detection model by yourself"""

    def __init__(
            self, n_classes,
            in_module=None, backbone=None, neck=None, head=None,
            in_module_config=dict(), backbone_config=dict(),
            neck_config=dict(), head_config=dict(),
            **kwargs
    ):
        super().__init__()
        self.input = in_module(**in_module_config)
        self.backbone = backbone(**backbone_config)
        self.neck = neck(**neck_config)
        self.head = head(**head_config)


class BaseSemanticSegmentationModel(nn.Module):
    def forward(self, x, pix_images=None):
        if self.training and pix_images is not None:
            return dict(
                preds=x,
                loss=self.loss(x, pix_images)
            )
        else:
            return self.post_process(x)

    def post_process(self, preds):
        return preds.argmax(1)

    def loss(self, preds, pix_images):
        """
        Args:
            preds: [b, out_features + 1, h, w]
            pix_images: [b, h, w]

        """
        # value=255 is the padding or edge areas
        return F.cross_entropy(preds, pix_images, ignore_index=255)


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

        # in_ch -> out_ch
        # input_size -> output_size
        super().__init__(
            nn.Conv2d(in_ch, out_ch, (input_size - output_size) + 1)
        )


class OutModule(nn.Sequential):
    def __init__(self, out_features, in_features=1000):
        assert out_features <= in_features, f'output features must not be greater than {in_features}'
        super().__init__(
            nn.Linear(in_features, out_features)
        )


class Conv(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s=1, p=None, bias=False,
                 is_act=True, act=None, is_norm=True, norm=None, mode='cna',
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
                 is_act=True, act=None, is_norm=True, norm=None, mode='cna', only_upsample=False,
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

        super().__init__(*layers)

    @staticmethod
    def auto_p(k, s):
        """auto pad to divisible totally
        o=si+k-s-2p -> p=(k-s)/2
        e.g.
            input_size=224, k=4, s=2 if output_size=224*s=448, p=(k-s)/2=1
        """
        return int(np.ceil((k - s) / 2)) if k > s else 0


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps=1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class Linear(nn.Sequential):
    def __init__(self, in_features, out_features,
                 is_act=True, act=None, is_norm=True, bn=None, is_drop=False, drop_prob=0.7):
        self.is_act = is_act
        self.is_norm = is_norm
        self.is_drop = is_drop

        layers = []

        if self.is_drop:
            layers.append(nn.Dropout(drop_prob))

        layers.append(nn.Linear(in_features, out_features))

        if self.is_norm:
            layers.append(bn or nn.BatchNorm1d(out_features))

        if self.is_act:
            layers.append(act or nn.Sigmoid())

        self.out_features = out_features
        super().__init__(*layers)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

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
