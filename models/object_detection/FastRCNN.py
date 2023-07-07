import torch
from torch import nn
from utils.layers import Conv, ConvInModule
from . import GetBackbone

in_module_config = dict(
    in_ch=3,
    input_size=500,
)


class FastRCNN(nn.Module):
    def __init__(
            self, output_size,
            in_module=None, backbone=None, neck=None, head=None,
            in_module_config=in_module_config, backbone_config=dict(),
            neck_config=dict(), head_config=dict(),
            **kwargs
    ):
        super().__init__()

        self.input = in_module(**in_module_config) if in_module else ConvInModule(**in_module_config)
        if backbone is None:
            self.backbone = GetBackbone.get_mobilenet_v2()
        elif isinstance(backbone, str):
            self.backbone = GetBackbone.get_one(backbone, backbone_config)
        else:
            self.backbone = backbone
        self.neck = nn.Module()
        self.head = nn.Module()

    def forward(self, images):
        features = self.input(images)
        features = self.backbone(features)

        if isinstance(features, torch.Tensor):
            features = [features]


class SPP(nn.Module):
    """[Spatial pyramid pooling in deep convolutional networks for visual recognition](https://arxiv.org/pdf/1406.4729.pdf)"""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1)
        self.pool_list = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1)

    def forward(self, x):
        x = self.cv1(x)
        x = torch.cat([x] + [pool(x) for pool in self.pool_list], dim=1)
        x = self.cv2(x)
        return x


Model = FastRCNN
