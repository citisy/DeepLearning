import warnings

from torch import nn
from ..layers import ConvInModule, Linear, OutModule
from utils import op_utils

make_backbone_fn = op_utils.RegisterTables()


class BaseImgClsModel(nn.Module):
    """a template to make an image classifier model"""

    def __init__(
            self,
            in_ch=3, input_size=None, out_features=None,
            in_module=None, backbone=None, neck=None, head=None, out_module=None,
            backbone_in_ch=None, backbone_input_size=None, head_hidden_features=1000, drop_prob=0.1,
            is_multi_label=False
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
        if is_multi_label:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

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
        """
        Args:
            pred_label: (b, n_features) in [-inf, +inf] without activation
            true_label:
                if multi_label: (b, n_labels) in {0, 1}
                if not multi_label: (b, ) in {0, 1, ..., n_labels-1}

        """
        return self.criterion(pred_label, true_label)


def init_register():
    import os
    import glob

    __all__ = []
    modules = glob.glob(os.path.dirname(__file__) + "/*.py")

    for module in modules:
        module_name = os.path.basename(module)[:-3]
        if module_name != '__init__':
            try:
                exec(f"from .{module_name} import *")
                __all__.append(module_name)
            except Exception as e:
                warnings.warn(f"import {module_name} error: {e}")
