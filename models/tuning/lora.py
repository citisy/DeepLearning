import torch
from torch import nn
import warnings
from utils import torch_utils


class LoraModule(nn.Module):
    def forward(self, x):
        with torch.no_grad():
            y = super().forward(x)

        x = x.view(-1, self.A.shape[0])
        return y + torch.mm(torch.mm(x, self.A), self.B)


class Linear(LoraModule, nn.Linear):
    def __init__(self, base_layer: nn.Linear, r=8):
        super().__init__(
            base_layer.in_features,
            base_layer.out_features,
            bias=base_layer.bias is not None
        )

        self.load_state_dict(base_layer.state_dict())

        self.A = nn.Parameter(torch.randn(self.in_features, r))
        self.B = nn.Parameter(torch.zeros(r, self.out_features))


class Conv1D(LoraModule, nn.Conv1d):
    def __init__(self, base_layer: nn.Conv1d, r=8):
        super().__init__(
            base_layer.in_channels,
            base_layer.out_channels,
            base_layer.kernel_size,
            groups=base_layer.groups,
            bias=base_layer.bias is not None
        )

        self.load_state_dict(base_layer.state_dict())

        self.A = nn.Parameter(torch.randn(self.in_channels, r))
        self.B = nn.Parameter(torch.zeros(r, self.out_channels))


class Embedding(LoraModule, nn.Embedding):
    def __init__(self, base_layer: nn.Embedding, r=8):
        super().__init__(
            base_layer.num_embeddings,
            base_layer.embedding_dim,
        )

        self.load_state_dict(base_layer.state_dict())

        self.A = nn.Parameter(torch.randn(self.num_embeddings, r))
        self.B = nn.Parameter(torch.zeros(r, self.embedding_dim))


class ModelWarp:
    """
    Usages:
        .. code-block:: python

            model = ...
            model = ModelWarp(include=['to_qkv']).warp(model)
            model.to(device)

            # your train step
            ...
    """

    layer_mapping = {
        nn.Linear: Linear,
        nn.Conv1d: Conv1D,
        nn.Embedding: Embedding,
    }

    def __init__(self, include=(), exclude=(), r=8):
        self.include = include
        self.exclude = exclude
        self.r = r

    def warp(self, model: nn.Module):
        torch_utils.ModuleManager.freeze_module(model, allow_train=True)
        layers = torch_utils.ModuleManager.get_module_by_name(model, include=self.include, exclude=self.exclude)
        if len(layers) == 0:
            warnings.warn(f'can not find any layer by include={self.include} and exclude={self.exclude}')

        for current_m, name, full_name in layers:
            new = self.get_new_module(getattr(current_m, name))
            if new:
                setattr(current_m, name, new)
            else:
                warnings.warn(f'not support lora layer type for {full_name}')

        return model

    def get_new_module(self, module: nn.Module):
        for out, new in self.layer_mapping.items():
            if isinstance(module, out):
                return new(module, r=self.r)
