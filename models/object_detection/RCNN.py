from torch import nn
from ..layers import ConvInModule, OutModule


class ObjectDetectModelExample(nn.Module):
    def __init__(
            self,
            in_ch=None, input_size=None, output_size=None,
            in_module=None, out_module=None,
            **kwargs
    ):
        super().__init__()

        if in_module is None:
            in_module = ConvInModule(in_ch, input_size, out_ch=3, output_size=224)

        if out_module is None:
            out_module = OutModule(output_size, in_features=1000)

        self.input = nn.Module()
        self.backbone = nn.Module()
        self.neck = nn.Module()
        self.head = nn.Module()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.neck(x)

        return x


class SelectiveSearch(nn.Module):
    def __init__(self):
        super().__init__()
