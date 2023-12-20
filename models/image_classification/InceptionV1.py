import torch
from torch import nn
from . import BaseImgClsModel
from ..layers import Conv, Linear, ConvInModule, OutModule

# from table 1
# (1*1 out_ch, 3*3 in_ch, 3*3 out_ch, 5*5 in_ch, 5*5 out_ch, pool out_ch)
Inception_config = (
    (64, 96, 128, 16, 32, 32),  # 3a, 256
    (128, 128, 192, 32, 96, 64),  # 3b, 480

    (192, 96, 208, 16, 48, 64),  # 4a, 512
    (160, 112, 224, 24, 64, 64),  # 4b, 512
    (128, 128, 256, 24, 64, 64),  # 4c, 512
    (112, 144, 288, 32, 64, 64),  # 4d, 512
    (256, 160, 320, 32, 128, 128),  # 4e, 528

    (256, 160, 320, 32, 128, 128),  # 5a, 832
    (384, 192, 384, 48, 128, 128)  # 5b, 1024
)


class Inception(BaseImgClsModel):
    def __init__(
            self,
            in_ch=None, input_size=None, out_features=None,
            backbone=None,
            is_norm=True, backbone_config=Inception_config, **kwargs
    ):
        assert len(backbone_config) == 9, f'Must have 9 Inception blocks, but have {len(backbone_config)} Inception blocks now'

        backbone = backbone or Backbone
        backbone = backbone(backbone_config=backbone_config, is_norm=is_norm)

        super().__init__(
            in_ch=in_ch,
            input_size=input_size,
            out_features=out_features,
            backbone=backbone,
            **kwargs
        )


class Model(Inception):
    """[Going deeper with convolutions](https://arxiv.org/pdf/1409.4842v1.pdf)
    See Also `torchvision.models.inception`
    """

    def __init__(
            self, in_ch=None, input_size=None, out_features=None,
            backbone_config=Inception_config, **kwargs
    ):
        super().__init__(in_ch, input_size, out_features, backbone_config=backbone_config,
                         is_norm=False, **kwargs)


class Backbone(nn.Module):
    def __init__(self, backbone_config=Inception_config, is_norm=True):
        super().__init__()
        layers = [
            Conv(3, 64, 7, s=2, is_norm=is_norm),
            nn.MaxPool2d(3, 2, padding=1),

            Conv(64, 192, 3, is_norm=is_norm),
            nn.MaxPool2d(3, 2, padding=1),
        ]

        in_ch = 192

        for out_ch in backbone_config[0:2]:
            layers.append(InceptionA(in_ch, *out_ch))
            in_ch = out_ch[0] + out_ch[2] + out_ch[4] + out_ch[5]

        layers.append(nn.MaxPool2d(3, 2, padding=1))

        for out_ch in backbone_config[2:7]:
            layers.append(InceptionA(in_ch, *out_ch))
            in_ch = out_ch[0] + out_ch[2] + out_ch[4] + out_ch[5]

        layers.append(nn.MaxPool2d(3, 2, padding=1))

        for out_ch in backbone_config[7:9]:
            layers.append(InceptionA(in_ch, *out_ch))
            in_ch = out_ch[0] + out_ch[2] + out_ch[4] + out_ch[5]

        self.conv_seq = nn.ModuleList(layers)
        self.out_channels = in_ch

    def forward(self, x):
        for layer in self.conv_seq:
            x = layer(x)
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
