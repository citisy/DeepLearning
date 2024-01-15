import copy
import torch
from torch import nn, optim
import torch.nn.functional as F
from ..layers import Conv, Linear
from ..image_classification import BaseImgClsModel
from .siamese import Block

default_config = dict(
    # (out_ch, k, s)
    conv_config=[(96, 7, 3), 'p', (192, 5), 'p', (256, 3)]
)
deep_config = dict(
    # from https://github.com/szagoruyko/cvpr15deepcompare/tree/master/training/models/2chdeep.lua
    conv_config=[(96, 4, 3), *[(96, 3)] * 3, 'p', *[(192, 3)] * 3]
)


class Model(BaseImgClsModel):
    """refer to
    paper:
        - [Learning to Compare Image Patches via Convolutional Neural Networks](https://arxiv.org/pdf/1504.03641.pdf)
    code:
        - https://github.com/szagoruyko/cvpr15deepcompare/tree/master/training/models/2chavg.lua
    """

    def __init__(self, backbone_config=None, **kwargs):
        if backbone_config is None:
            backbone_config = default_config

        backbone = Block(2, **backbone_config)
        neck = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten()
        )
        in_features = backbone.out_channels * 2 * 2
        head = nn.Sequential(
            Linear(in_features, in_features, mode='la', act=nn.ReLU()),
            nn.Linear(in_features, 1)  # pos to 1 and neg to -1
        )

        super().__init__(
            in_module=nn.Identity(),  # placeholder
            backbone=backbone,
            neck=neck,
            head=head,
            **kwargs
        )

    def loss(self, pred_label, true_label):
        return F.hinge_embedding_loss(pred_label, true_label)


class Backbone(nn.Module):
    def __init__(self, conv_config, two_stream=True):
        super().__init__()
        self.two_stream = two_stream
        self.block = Block(2, conv_config)

        # Central-surround two-stream network
        if two_stream:
            self.eye_block = copy.deepcopy(self.block)

    def forward(self, x1, x2=None):
        x = self.block(x1)

        if self.two_stream:
            _x = self.eye_block(x2)
            x = torch.cat((x, _x), 1)

        return x
