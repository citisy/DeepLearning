from .InceptionV1 import Inception, Inception_config


class InceptionV2(Inception):
    """[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
    See Also `torchvision.models.inception`
    """

    def __init__(
            self, in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            backbone_config=Inception_config, drop_prob=0.4
    ):
        super().__init__(in_ch, input_size, output_size, in_module, out_module,
                         is_norm=True, backbone_config=backbone_config, drop_prob=drop_prob)


Model = InceptionV2
