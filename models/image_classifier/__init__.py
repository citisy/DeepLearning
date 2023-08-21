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

        from ..image_classifier.ResNet import Backbone, Res34_config

        default_config = default_config or dict(backbone_config=Res34_config)
        backbone = Backbone(**default_config)
        return backbone

    @staticmethod
    def get_resnet50():
        from ..image_classifier.ResNet import Backbone, Res50_config

        backbone = Backbone(**Res50_config)
        return backbone

    @staticmethod
    def get_resnet101():
        from ..image_classifier.ResNet import Backbone, Res101_config

        backbone = Backbone(**Res101_config)
        return backbone

    @staticmethod
    def get_mobilenet(default_config=dict()):
        from ..image_classifier.MobileNetV1 import Backbone, default_config as config

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
        from ..image_classifier.VGG import Backbone, VGG19_config

        default_config = default_config or dict(backbone_config=VGG19_config)
        backbone = Backbone(**default_config)
        return backbone
