from .InceptionV1 import Inception, Inception_config


class Model(Inception):
    """[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
    See Also `torchvision.models.inception`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, is_norm=True, **kwargs)

