import torch
from torch import nn
from ..layers import Conv, Cache

# in_ch, (n_conv per C3 block,), (cache_block_idx, )
darknet_config = (64, (3, 6, 9, 3), (1, 2))


class Backbone(nn.Module):
    """refer to https://github.com/ultralytics/yolov5"""

    def __init__(self, in_ch=3, backbone_config=darknet_config):
        super().__init__()

        out_ch, n_conv, cache_block_idx = backbone_config

        layers = [
            Conv(in_ch, out_ch, 6, s=2, p=2, act=nn.SiLU())
        ]

        in_ch = out_ch

        for i, n in enumerate(n_conv):
            out_ch = in_ch * 2
            m = nn.Sequential(
                Conv(in_ch, out_ch, 3, 2, act=nn.SiLU()),
                C3(out_ch, out_ch, n=n)
            )

            layers.append(m)
            if i in cache_block_idx:
                layers.append(Cache())

            in_ch = out_ch

        layers.append(SPPF(in_ch, in_ch, k=5))

        self.conv_list = nn.ModuleList(layers)
        self.out_channels = in_ch

    def forward(self, x):
        features = []
        for m in self.conv_list:
            if isinstance(m, Cache):
                x, features = m(x, features)
            else:
                x = m(x)

        features.append(x)

        return features


class C3(nn.Module):
    # note that, to distinguish C3 and ResBlock.
    # C3 gives y = a(cat(x, f(x)))
    # and ResBlock gives y = a(x + f(x))
    def __init__(self, in_ch, out_ch, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_ch = int(out_ch * e)
        self.seq1 = nn.Sequential(
            Conv(in_ch, hidden_ch, 1, 1, act=nn.SiLU()),
            *(Bottleneck(hidden_ch, hidden_ch, shortcut, g, e=1.0) for _ in range(n))
        )
        self.cv2 = Conv(in_ch, hidden_ch, 1, 1, act=nn.SiLU())
        self.cv3 = Conv(hidden_ch * 2, out_ch, 1, act=nn.SiLU())  # optional act=FReLU(c2)

    def forward(self, x):
        x1 = self.seq1(x)
        x2 = self.cv2(x)
        x = torch.cat((x1, x2), 1)
        return self.cv3(x)


class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_ch * e)  # hidden channels
        self.cv1 = Conv(in_ch, c_, 1, 1, act=nn.SiLU())
        self.cv2 = Conv(c_, out_ch, 3, 1,  act=nn.SiLU(), groups=g)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, in_ch, out_ch, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = in_ch // 2  # hidden channels
        self.cv1 = Conv(in_ch, c_, 1, 1, act=nn.SiLU())
        self.cv2 = Conv(c_ * 4, out_ch, 1, 1, act=nn.SiLU())
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

