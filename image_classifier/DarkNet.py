import torch
from torch import nn
from utils.layers import Conv, Linear, ConvInModule, OutModule

# in_ch, (n_conv per C3 block,), (cache_block, )
darknet_config = (64, (3, 6, 9, 3), (1, 2))


class CspDarkNet(nn.Module):
    """refer to https://github.com/ultralytics/yolov5"""
    def __init__(self, in_ch=3, conv_config=darknet_config):
        super().__init__()

        out_ch, n_convs, cache_block = conv_config

        layers = [
            Conv(in_ch, out_ch, 6, s=2, p=2)
        ]

        in_ch = out_ch

        for i, n_conv in enumerate(n_convs):
            out_ch = in_ch * 2
            m = nn.Sequential(
                Conv(in_ch, out_ch, 3, 2),
                C3Block(out_ch, out_ch)
            )

            layers.append(m)
            if i in cache_block:
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


class C3Block(nn.Module):
    def __init__(self, in_ch, out_ch, n_block=3, **C3_kwargs):
        super().__init__()

        self.m = nn.Sequential(
            C3(in_ch, out_ch, **C3_kwargs),
            *[C3(out_ch, out_ch, **C3_kwargs) for _ in range(n_block - 1)],
        )

    def forward(self, x):
        return self.m(x)


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, in_ch, out_ch, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_ch * e)  # hidden channels
        self.cv1 = Conv(in_ch, c_, 1, 1)
        self.cv2 = Conv(in_ch, c_, 1, 1)
        self.cv3 = Conv(2 * c_, out_ch, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_ch, out_ch, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_ch * e)  # hidden channels
        self.cv1 = Conv(in_ch, c_, 1, 1)
        self.cv2 = Conv(c_, out_ch, 3, 1, conv_kwargs=dict(groups=g))
        self.add = shortcut and in_ch == out_ch

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, in_ch, out_ch, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = in_ch // 2  # hidden channels
        self.cv1 = Conv(in_ch, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, out_ch, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Cache(nn.Module):
    def __init__(self, idx=None):
        super().__init__()
        self.idx = idx

    def forward(self, x, features):
        if self.idx is not None:
            features[self.idx] = x
        else:
            features.append(x)
        return x, features


