import torch
from torch import nn
from torch.nn import functional as F
from . import BaseSemSegModel
from ..layers import Conv, Linear, ConvInModule, OutModule, ConvT
from ..image_classification.ResNet import Backbone, Config
from utils.torch_utils import ModuleManager


class Model(BaseSemSegModel):
    """http://arxiv.org/pdf/1411.4038
    see also `torchvision.models.segmentation.fcn.fcn_resnet50`
    """

    def __init__(
            self,
            in_ch=None, input_size=None, out_features=None, in_module=None, backbone=None,
            backbone_in_ch=None, backbone_input_size=None,
    ):
        super().__init__()
        self.input = in_module if in_module is not None else ConvInModule(in_ch, input_size, out_ch=backbone_in_ch, output_size=backbone_input_size)

        # self.backbone = Backbone(VGG16_config)
        if backbone is None:
            config = Config.resnet50_backbone[:3]
            self.backbone = Backbone(self.input.out_channels, backbone_config=config)
        else:
            self.backbone = backbone

        # get stride
        x = torch.rand((1, 3, input_size, input_size))
        x = self.backbone(x)
        stride = int(input_size / x.shape[-1])
        f = stride * 4
        assert input_size % f == 0, f"{input_size = } must be a multiple of {f}"

        # background is the addition class
        self.out_features = out_features + 1
        self.neck = Neck(self.backbone.out_channels, self.out_features)
        self.head = Head(self.out_features, upsample_ratio=stride * 2)
        ModuleManager.initialize_layers(self)

    def forward(self, x, label_masks=None):
        x = self.backbone(x)
        x, features = self.neck(x)
        x = self.head(x, features)
        return super().forward(x, label_masks)


class Neck(nn.Module):
    def __init__(self, in_ch, out_features, n_layers=3):
        super().__init__()
        out_ch = in_ch
        layers1 = []
        layers2 = []
        for _ in range(n_layers):
            out_ch //= 2
            layers1.append(Conv(in_ch, out_ch, 3, s=2))
            layers2.append(Conv(out_ch, out_features, 1))
            in_ch = out_ch

        self.conv_list1 = nn.ModuleList(layers1)
        self.conv_list2 = nn.ModuleList(layers2)

    def forward(self, x):
        features = []
        for m1, m2 in zip(self.conv_list1, self.conv_list2):
            x = m1(x)
            features.append(m2(x))
        return x, features


class Head(nn.Module):
    def __init__(self, out_features, upsample_ratio, n_layers=3):
        super().__init__()
        k = 4
        layers = []
        for n in range(n_layers):
            if n < n_layers - 1:
                conv = ConvT(out_features, out_features, k, s=2, bias=False, is_norm=False, is_act=False)
            else:
                # note that, k must be 2 times of s
                conv = ConvT(out_features, out_features, upsample_ratio * 2, s=upsample_ratio, bias=False, is_norm=False, is_act=False)
            layers.append(conv)
            k *= 2

        self.conv_list = nn.ModuleList(layers)

    def forward(self, x, features):
        features = features[::-1]
        x = features[0]
        for i, m in enumerate(self.conv_list):
            x = m(x)
            if i < len(features) - 1:
                x += features[i + 1]

        return x
