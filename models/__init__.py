import torch
from torch import nn


class BaseImgClsModel(nn.Module):
    """a template to make a image classifier model by yourself"""

    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None, backbone=None,
            backbone_config=None):
        super().__init__()
        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=224)

        if out_module is None:
            out_module = OutModule(output_size, input_size=1000)

        self.input = in_module
        self.backbone = backbone(backbone_config=backbone_config)
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fcn = nn.Sequential(
            Linear(1 * 1 * self.backbone.out_channels, 1000),
            out_module
        )

    def forward(self, x):
        x = self.input(x)
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fcn(x)

        return x


class BaseObjectDetectionModel(nn.Module):
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

    def forward(self, x):
        x = self.input(x)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return x


class SimpleInModule(nn.Sequential):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)


class ConvInModule(nn.Module):
    def __init__(self, in_ch=3, input_size=224, out_ch=None, output_size=None):
        super().__init__()

        out_ch = out_ch or in_ch
        output_size = output_size or input_size

        assert in_ch <= out_ch, f'input channel must not be greater than {out_ch = }'
        assert input_size >= output_size, f'input size must not be smaller than {output_size = }'

        self.in_channels = in_ch
        self.out_channels = out_ch
        self.input_size = input_size

        # in_ch -> min_in_ch
        # input_size -> min_input_size
        self.layer = Conv(in_ch, out_ch, (input_size - output_size) + 1, p=0, is_norm=False)

    def forward(self, x):
        return self.layer(x)


class OutModule(nn.Module):
    def __init__(self, output_size, input_size=1000):
        super().__init__()

        assert output_size <= input_size, f'output size must not be greater than {input_size}'

        self.layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.layer(x)


class Conv(nn.Module):
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
        super().__init__()
        self.is_act = is_act
        self.is_norm = is_norm
        self.in_channels = in_ch
        self.out_channels = out_ch

        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]

        layers = []

        for m in mode:
            if m == 'c':
                layers.append(nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias, **conv_kwargs))
            elif m == 'n' and is_norm:
                layers.append(norm or nn.BatchNorm2d(out_ch))
            elif m == 'a' and is_act:
                layers.append(act or nn.ReLU(True))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class ConvT(nn.Module):
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
        super().__init__()
        self.is_act = is_act
        self.is_norm = is_norm
        self.in_channels = in_ch
        self.out_channels = out_ch

        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]

        layers = []

        for m in mode:
            if m == 'c':
                layers.append(nn.ConvTranspose2d(in_ch, out_ch, k, s, p, bias=bias, **conv_kwargs))
            elif m == 'n' and is_norm:
                layers.append(norm or nn.BatchNorm2d(out_ch))
            elif m == 'a' and is_act:
                layers.append(act or nn.ReLU(True))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class Linear(nn.Module):
    def __init__(self, in_size, out_size,
                 is_act=True, act=None, is_bn=True, bn=None, is_drop=False, drop_prob=0.7):
        super().__init__()
        self.is_act = is_act
        self.is_bn = is_bn
        self.is_drop = is_drop

        layers = [nn.Linear(in_size, out_size)]

        if self.is_bn:
            layers.append(bn or nn.BatchNorm1d(out_size))
        if self.is_drop:
            layers.append(nn.Dropout(drop_prob))
        if self.is_act:
            layers.append(act or nn.Sigmoid())

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)
