from torch import nn
import torch.nn.functional as F
from utils.torch_utils import ModuleManager
from ..layers import ConvInModule, Linear, OutModule


class BaseImgClsModel(nn.Module):
    """a template to make a image classifier model by yourself"""

    def __init__(
            self,
            in_ch=3, input_size=None, out_features=None,
            in_module=None, backbone=None, neck=None, head=None, out_module=None,
            backbone_in_ch=None, backbone_input_size=None, head_hidden_features=1000, drop_prob=0.4
    ):
        super().__init__()

        self.in_channels = in_ch
        self.input_size = input_size
        self.out_features = out_features

        # `bool(nn.Sequential()) = False`, so do not ues `input = in_module or ConvInModule()`
        # if in_module is None, in_ch and input_size must be set
        self.input = in_module if in_module is not None else ConvInModule(in_ch, input_size, out_ch=backbone_in_ch, output_size=backbone_input_size)
        self.backbone = backbone
        self.neck = neck if neck is not None else nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.head = head if head is not None else nn.Sequential(
            Linear(self.backbone.out_channels, head_hidden_features, mode='dla', drop_prob=drop_prob),
            out_module or OutModule(out_features, in_features=head_hidden_features)
        )

        ModuleManager.initialize_layers(self)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.fit(*args, **kwargs)
        else:
            return self.inference(*args, **kwargs)

    def fit(self, x, true_label=None):
        x = self.process(x)
        loss = self.loss(pred_label=x, true_label=true_label)
        return {'pred': x, 'loss': loss}

    def inference(self, x):
        x = self.process(x)
        return {'pred': x}

    def process(self, x):
        x = self.input(x)
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def loss(self, pred_label, true_label):
        return F.cross_entropy(pred_label, true_label)


class GetBackbone:
    @classmethod
    def get_one(cls, name='ResNet', default_config=dict()):
        backbones = {
            'Vgg': cls.get_vgg,
            'ResNet': cls.get_resnet,
            'MobileNet': cls.get_mobilenet,
            'MobileNetV2': cls.get_mobilenet_v2
        }

        return backbones[name](default_config)

    @staticmethod
    def get_resnet(default_config=dict()):
        # from torchvision.models.resnet import BasicBlock, ResNet as Model
        # model = Model(BasicBlock, [2, 2, 2, 2])
        # backbone = ...   # todo
        # backbone.out_channels = 1280

        from .ResNet import Backbone, Res34_config

        default_config = default_config or dict(backbone_config=Res34_config)
        backbone = Backbone(**default_config)
        return backbone

    @staticmethod
    def get_resnet50():
        from .ResNet import Backbone, Res50_config

        backbone = Backbone(**Res50_config)
        return backbone

    @staticmethod
    def get_resnet101():
        from .ResNet import Backbone, Res101_config

        backbone = Backbone(**Res101_config)
        return backbone

    @staticmethod
    def get_mobilenet(default_config=dict()):
        from .MobileNetV1 import Backbone, default_config as config

        default_config = default_config or dict(backbone_config=config)
        backbone = Backbone(**default_config)
        return backbone

    @staticmethod
    def get_mobilenet_v2(default_config=dict()):
        from torchvision.models.mobilenet import MobileNetV2 as Model

        backbone = Model(**default_config).features
        backbone.out_channels = 1280
        return backbone

    @staticmethod
    def get_vgg(default_config=dict()):
        from .VGG import Backbone, VGG19_config

        default_config = default_config or dict(backbone_config=VGG19_config)
        backbone = Backbone(**default_config)
        return backbone
