import copy
import torch
from torch import nn, optim
import torch.nn.functional as F
from ..layers import Conv, Linear
from ..image_classification import BaseImgClsModel
from ..image_classification.CspDarkNet import SPPF

default_config = dict(
    # (out_ch, k, s)
    conv_config=[(96, 7, 3), 'p', (192, 5), 'p', (256, 3)]
)


class Model(BaseImgClsModel):
    """refer to
    paper:
        - Learning a similarity metric discriminatively, with application to face verification.
        - [Learning to Compare Image Patches via Convolutional Neural Networks](https://arxiv.org/pdf/1504.03641.pdf)
    code:
        - https://github.com/szagoruyko/cvpr15deepcompare/tree/master/training/models/siam.lua
    """

    def __init__(self, backbone_config=None, **kwargs):
        if backbone_config is None:
            backbone_config = default_config

        backbone = Backbone(**backbone_config)
        in_features = backbone.out_channels
        head = nn.Sequential(
            Linear(in_features, in_features, mode='la', act=nn.ReLU()),
            nn.Linear(in_features, 1)  # pos to 1 and neg to -1
        )

        super().__init__(
            in_module=nn.Identity(),  # placeholder
            backbone=backbone,
            head=head,
            **kwargs
        )

    def loss(self, pred_label, true_label):
        return F.hinge_embedding_loss(pred_label, true_label)


class Backbone(nn.Module):
    def __init__(self, conv_config, shared=True, two_stream=True):
        super().__init__()
        self.two_stream = two_stream

        if shared:
            self.block = SharedBlock(conv_config)
        else:
            self.block = UnsharedBlock(conv_config)

        # Central-surround two-stream network
        if two_stream:
            self.eye_block = copy.deepcopy(self.block)

    def forward(self, x1, x2, x3=None, x4=None):
        x = self.block(x1, x2)

        if self.two_stream:
            _x = self.eye_block(x3, x4)
            x = torch.cat((x, _x), 1)

        return x


class SharedBlock(nn.Module):
    def __init__(self, conv_config):
        super().__init__()
        self.branch = Block(1, conv_config)
        self.out_channels = self.branch.out_channels * 2

    def forward(self, x1, x2):
        x1 = self.branch(x1)
        x2 = self.branch(x2)

        return torch.cat((x1, x2), 1)


class UnsharedBlock(nn.Module):
    """pseudo-siamese"""

    def __init__(self, conv_config):
        super().__init__()
        self.branch1 = Block(1, conv_config)
        self.branch2 = Block(1, conv_config)
        self.out_channels = self.branch.out_channels * 2

    def forward(self, x1, x2):
        x1 = self.branch1(x1)
        x2 = self.branch2(x2)

        return torch.cat((x1, x2), 1)


class Block(nn.Sequential):
    def __init__(self, in_ch, conv_config):
        layers = []
        for i, args in enumerate(conv_config):
            if args == 'p':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.append(Conv(in_ch, *args, mode='ca'))
                in_ch = args[0]

        layers.append(SPPF(in_ch, in_ch))
        self.out_channels = in_ch
        super().__init__(*layers)
