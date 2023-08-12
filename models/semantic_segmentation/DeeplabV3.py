import torch
from torch import nn
from torch.nn import functional as F
from ..layers import Conv, Linear, ConvInModule, OutModule, ConvT, BaseSemanticSegmentationModel
# from ..image_classifier.VGG import Backbone, VGG16_config
from ..image_classifier.ResNet import Backbone, Res50_config, Res101_config
from utils.torch_utils import initialize_layers

ASPP_config = dict(
    out_ch=256,
    dilations=(12, 24, 36),
    only_upsample=True,
)


class Model(BaseSemanticSegmentationModel):
    """
    see also `torchvision.models.segmentation.deeplabv3.deeplabv3_resnet50`
    """
    def __init__(
            self,
            in_ch=None, input_size=None, out_features=None, in_module=None, backbone=None,
            backbone_in_ch=None, backbone_input_size=None, neck_config=ASPP_config,
    ):
        super().__init__()
        self.input = in_module if in_module is not None else ConvInModule(in_ch, input_size, out_ch=backbone_in_ch, output_size=backbone_input_size)

        if backbone is None:
            config = Res50_config[:3]
            self.backbone = Backbone(self.input.out_channels, backbone_config=config)
        else:
            self.backbone = backbone

        x = torch.rand((1, 3, input_size, input_size))
        x = self.backbone(x)
        stride = int(input_size / x.shape[-1])
        assert input_size % stride == 0, f"{input_size = } must be a multiple of {stride}"

        self.out_features = out_features + 1
        self.neck = ASPP(self.backbone.out_channels, x.shape[-1], **neck_config)
        self.head = nn.Sequential(
            Conv(self.neck.out_channels, self.neck.out_channels, 3),
            Conv(self.neck.out_channels, self.out_features, 1, is_norm=False, is_act=False),
            ConvT(self.out_features, self.out_features, k=stride * 2, s=stride, is_norm=False, is_act=False)
        )
        initialize_layers(self)

    def forward(self, x, pix_images=None):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return super().forward(x, pix_images)


class ASPP(nn.Module):
    def __init__(self, in_ch, input_size, out_ch=256, dilations=(12, 24, 36), only_upsample=True):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch

        layers = [Conv(in_ch, out_ch, 1)]

        for dilation in dilations:
            layers.append(Conv(in_ch, out_ch, 3, p=dilation, dilation=dilation))

        layers.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(in_ch, out_ch, 1),
            ConvT(out_ch, out_ch, input_size * 2, s=input_size, only_upsample=only_upsample),
        ))

        self.conv_list = nn.ModuleList(layers)

        self.head = nn.Sequential(
            Conv(len(self.conv_list) * out_ch, out_ch, 1),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        ys = []
        for conv in self.conv_list:
            ys.append(conv(x))
        x = torch.cat(ys, dim=1)
        return self.head(x)
