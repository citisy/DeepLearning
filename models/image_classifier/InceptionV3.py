import torch
from torch import nn
from .. import Conv, Linear, ConvInModule, OutModule

# from torchvision.models.inception.inception_v3
# (1*1 out_ch, 3*3 in_ch, 3*3 out_ch, 5*5 in_ch, 5*5 out_ch, pool out_ch)
InceptionV3_config = (
    (64, 48, 64, 64, 96, 64),  # 64+64+96+64=288
    (64, 48, 64, 64, 96, 64),  # 288
    (64, 384, 384, 64, 96, 224),  # 758

    (192, 128, 192, 128, 192, 192),  # 192+192+192+192=768
    (192, 160, 192, 160, 192, 192),  # 768
    (192, 160, 192, 160, 192, 192),  # 768
    (192, 192, 192, 192, 192, 192),  # 768
    (192, 192, 320, 192, 192, 576),  # 1280

    (320, 384, 384, 448, 384, 192),  # 320+384*2+384*2+192=2048
    (320, 384, 384, 448, 384, 192),  # 2048
)


class InceptionV3(nn.Module):
    """[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)
    See Also `torchvision.models.inception.inception_v3`
    """

    def __init__(
            self, in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            backbone_config=InceptionV3_config, drop_prob=0.4
    ):
        super().__init__()

        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=299)

        if out_module is None:
            out_module = OutModule(output_size, input_size=1000)

        assert len(backbone_config) == 10, f'Must have 9 Inception blocks, but have {len(backbone_config)} Inception blocks now'

        self.input = in_module
        self.backbone = Backbone(backbone_config=backbone_config)
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fcn = nn.Sequential(
            nn.Dropout(drop_prob),
            Linear(self.backbone.out_channels, 1000),
            out_module,
        )

    def forward(self, x):
        x = self.input(x)
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.fcn(x)

        return x


class Backbone(nn.Module):
    def __init__(self, backbone_config=InceptionV3_config):
        super().__init__()

        layers = [
            Conv(3, 32, 3, s=2, p=0),
            Conv(32, 32, 3, p=0),
            Conv(32, 64, 3, p=1),
            nn.MaxPool2d(3, 2),
            Conv(64, 80, 3, p=0),
            Conv(80, 192, 3, s=2, p=0),
            Conv(192, 288, 3, p=0),
        ]

        in_ch = 288

        for out_ch in backbone_config[0:3]:
            layers.append(InceptionB(in_ch, *out_ch))
            in_ch = out_ch[0] + out_ch[2] + out_ch[4] + out_ch[5]

        layers.append(nn.MaxPool2d(3, 2, padding=1))

        for out_ch in backbone_config[3:8]:
            layers.append(InceptionC(in_ch, *out_ch))
            in_ch = out_ch[0] + out_ch[2] + out_ch[4] + out_ch[5]

        layers.append(nn.MaxPool2d(3, 2))

        for out_ch in backbone_config[8:10]:
            layers.append(InceptionD(in_ch, *out_ch))
            in_ch = out_ch[0] + out_ch[2] * 2 + out_ch[4] * 2 + out_ch[5]

        self.conv_seq = nn.Sequential(*layers)
        self.out_channels = in_ch

    def forward(self, x):
        return self.conv_seq(x)


class InceptionB(nn.Module):
    def __init__(self, in_ch, *out_ches, is_bn=True):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ches[0] + out_ches[2] + out_ches[4] + out_ches[5]

        self.conv1 = Conv(in_ch, out_ches[0], 1, is_norm=is_bn)

        self.conv3 = nn.Sequential(
            Conv(in_ch, out_ches[1], 1, is_norm=is_bn),
            Conv(out_ches[1], out_ches[2], 3, is_norm=is_bn),
        )

        self.conv5 = nn.Sequential(
            Conv(in_ch, out_ches[3], 1, is_norm=is_bn),
            Conv(out_ches[3], out_ches[4], 3, is_norm=is_bn),
            Conv(out_ches[4], out_ches[4], 3, is_norm=is_bn)
        )

        self.pool = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),
            Conv(in_ch, out_ches[5], 1, is_norm=is_bn)
        )

    def forward(self, x):
        tmp = [
            self.conv1(x),
            self.conv3(x),
            self.conv5(x),
            self.pool(x)
        ]

        x = torch.cat(tmp, dim=1)

        return x


class InceptionC(nn.Module):
    def __init__(self, in_ch, *out_ches, is_bn=True):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ches[0] + out_ches[2] + out_ches[4] + out_ches[5]

        self.conv1 = Conv(in_ch, out_ches[0], 1, is_norm=is_bn)

        self.conv3 = nn.Sequential(
            Conv(in_ch, out_ches[1], 1, is_norm=is_bn),
            Conv(out_ches[1], out_ches[1], (1, 7), is_norm=is_bn),
            Conv(out_ches[1], out_ches[2], (7, 1), is_norm=is_bn),
        )

        self.conv5 = nn.Sequential(
            Conv(in_ch, out_ches[3], 1, is_norm=is_bn),
            Conv(out_ches[3], out_ches[3], (1, 7), is_norm=is_bn),
            Conv(out_ches[3], out_ches[3], (7, 1), is_norm=is_bn),
            Conv(out_ches[3], out_ches[3], (1, 7), is_norm=is_bn),
            Conv(out_ches[3], out_ches[4], (7, 1), is_norm=is_bn),
        )

        self.pool = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),
            Conv(in_ch, out_ches[5], 1, is_norm=is_bn)
        )

    def forward(self, x):
        tmp = [
            self.conv1(x),
            self.conv3(x),
            self.conv5(x),
            self.pool(x)
        ]

        x = torch.cat(tmp, dim=1)

        return x


class InceptionD(nn.Module):
    def __init__(self, in_ch, *out_ches, is_bn=True):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ches[0] + out_ches[2] * 2 + out_ches[4] * 2 + out_ches[5]

        self.conv1 = Conv(in_ch, out_ches[0], 1, is_norm=is_bn)

        self.conv3_1 = Conv(in_ch, out_ches[1], 1, is_norm=is_bn)
        self.conv3_2 = Conv(out_ches[1], out_ches[2], (1, 3), is_norm=is_bn)
        self.conv3_3 = Conv(out_ches[2], out_ches[2], (3, 1), is_norm=is_bn)

        self.conv5_1 = nn.Sequential(
            Conv(in_ch, out_ches[3], 1, is_norm=is_bn),
            Conv(out_ches[3], out_ches[4], 3, is_norm=is_bn),
        )
        self.conv5_2 = Conv(out_ches[4], out_ches[4], (1, 3), is_norm=is_bn)
        self.conv5_3 = Conv(out_ches[4], out_ches[4], (3, 1), is_norm=is_bn)

        self.pool = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),
            Conv(in_ch, out_ches[5], 1, is_norm=is_bn)
        )

    def forward(self, x):
        y1 = self.conv1(x)

        y2 = self.conv3_1(x)
        y21 = self.conv3_2(y2)
        y22 = self.conv3_3(y2)

        y3 = self.conv5_1(x)
        y31 = self.conv5_2(y3)
        y32 = self.conv5_3(y3)

        y4 = self.pool(x)

        x = torch.cat([y1, y21, y22, y31, y32, y4], dim=1)

        return x


Model = InceptionV3
