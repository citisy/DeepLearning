import torch
from torch import nn
from utils.layers import Conv, Linear, ConvInModule, OutModule

# from table 1
# (1*1 out_ch, 3*3 in_ch, 3*3 out_ch, 5*5 in_ch, 5*5 out_ch, pool out_ch)
InceptionV1_config = (
    (64, 96, 128, 16, 32, 32),      # 3a, 256
    (128, 128, 192, 32, 96, 64),    # 3b, 480

    (192, 96, 208, 16, 48, 64),     # 4a, 512
    (160, 112, 224, 24, 64, 64),    # 4b, 512
    (128, 128, 256, 24, 64, 64),    # 4c, 512
    (112, 144, 288, 32, 64, 64),    # 4d, 512
    (256, 160, 320, 32, 128, 128),  # 4e, 528

    (256, 160, 320, 32, 128, 128),  # 5a, 832
    (384, 192, 384, 48, 128, 128)   # 5b, 1024
)

# from torchvision.models.inception.inception_v3
InceptionV3_config = (
    (64, 48, 64, 64, 96, 64),        # 64+64+96+64=288
    (64, 48, 64, 64, 96, 64),        # 288
    (64, 384, 384, 64, 96, 224),     # 758

    (192, 128, 192, 128, 192, 192),  # 192+192+192+192=768
    (192, 160, 192, 160, 192, 192),  # 768
    (192, 160, 192, 160, 192, 192),  # 768
    (192, 192, 192, 192, 192, 192),  # 768
    (192, 192, 320, 192, 192, 576),  # 1280

    (320, 384, 384, 448, 384, 192),  # 320+384*2+384*2+192=2048
    (320, 384, 384, 448, 384, 192),  # 2048
)


class Inception(nn.Module):
    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            is_bn=True,
            conv_config=InceptionV1_config, drop_prob=0.4
    ):
        super().__init__()

        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=224)

        if out_module is None:
            out_module = OutModule(output_size, input_size=1000)

        assert len(conv_config) == 9, f'Must have 9 Inception blocks, but have {len(conv_config)} Inception blocks now'

        layers = [
            Conv(3, 64, 7, s=2, is_norm=is_bn),
            nn.MaxPool2d(3, 2, padding=1),

            Conv(64, 192, 3, is_norm=is_bn),
            nn.MaxPool2d(3, 2, padding=1),
        ]

        in_ch = 192

        for out_ch in conv_config[0:2]:
            layers.append(InceptionA(in_ch, *out_ch))
            in_ch = out_ch[0] + out_ch[2] + out_ch[4] + out_ch[5]

        layers.append(nn.MaxPool2d(3, 2, padding=1))

        for out_ch in conv_config[2:7]:
            layers.append(InceptionA(in_ch, *out_ch))
            in_ch = out_ch[0] + out_ch[2] + out_ch[4] + out_ch[5]

        layers.append(nn.MaxPool2d(3, 2, padding=1))

        for out_ch in conv_config[7:9]:
            layers.append(InceptionA(in_ch, *out_ch))
            in_ch = out_ch[0] + out_ch[2] + out_ch[4] + out_ch[5]

        self.input = in_module
        self.conv_seq = nn.Sequential(*layers)
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.fcn = nn.Sequential(
            nn.Dropout(drop_prob),
            Linear(1 * 1 * 1024, 1000),
            out_module,
        )

    def forward(self, x):
        x = self.input(x)
        x = self.conv_seq(x)
        x = self.flatten(x)
        x = self.fcn(x)

        return x


class InceptionV1(Inception):
    """[Going deeper with convolutions](https://arxiv.org/pdf/1409.4842v1.pdf)
    See Also `torchvision.models.inception`
    """

    def __init__(
            self, in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            conv_config=InceptionV1_config, drop_prob=0.4
    ):
        super().__init__(in_ch, input_size, output_size, in_module, out_module,
                         is_bn=False, conv_config=conv_config, drop_prob=drop_prob)


class InceptionV2(Inception):
    """[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
    See Also `torchvision.models.inception`
    """

    def __init__(
            self, in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            conv_config=InceptionV1_config, drop_prob=0.4
    ):
        super().__init__(in_ch, input_size, output_size, in_module, out_module,
                         is_bn=True, conv_config=conv_config, drop_prob=drop_prob)


class InceptionV3(nn.Module):
    """[Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567.pdf)
    See Also `torchvision.models.inception.inception_v3`
    """

    def __init__(
            self, in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            conv_config=InceptionV3_config, drop_prob=0.4
    ):
        super().__init__()

        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=299)

        if out_module is None:
            out_module = OutModule(output_size, input_size=1000)

        assert len(conv_config) == 10, f'Must have 9 Inception blocks, but have {len(conv_config)} Inception blocks now'

        layers = [
            in_module,

            Conv(3, 32, 3, s=2, p=0),
            Conv(32, 32, 3, p=0),
            Conv(32, 64, 3, p=1),
            nn.MaxPool2d(3, 2),
            Conv(64, 80, 3, p=0),
            Conv(80, 192, 3, s=2, p=0),
            Conv(192, 288, 3, p=0),
        ]

        in_ch = 288

        for out_ch in conv_config[0:3]:
            layers.append(InceptionB(in_ch, *out_ch))
            in_ch = out_ch[0] + out_ch[2] + out_ch[4] + out_ch[5]

        layers.append(nn.MaxPool2d(3, 2, padding=1))

        for out_ch in conv_config[3:8]:
            layers.append(InceptionC(in_ch, *out_ch))
            in_ch = out_ch[0] + out_ch[2] + out_ch[4] + out_ch[5]

        layers.append(nn.MaxPool2d(3, 2))

        for out_ch in conv_config[8:10]:
            layers.append(InceptionD(in_ch, *out_ch))
            in_ch = out_ch[0] + out_ch[2] * 2 + out_ch[4] * 2 + out_ch[5]

        layers.append(nn.AdaptiveAvgPool2d(1))

        self.conv_seq = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self.fcn = nn.Sequential(
            nn.Dropout(drop_prob),
            Linear(in_ch, 1000),
            out_module,
        )

    def forward(self, x):
        x = self.conv_seq(x)
        x = self.flatten(x)
        x = self.fcn(x)

        return x


class InceptionA(nn.Module):
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
            Conv(out_ches[3], out_ches[4], 5, is_norm=is_bn)
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


Model = InceptionV1
