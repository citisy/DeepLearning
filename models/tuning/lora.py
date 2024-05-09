import torch
from torch import nn
import warnings
from utils import torch_utils
from collections import OrderedDict


class LoraModule(nn.Module):
    def __init__(self, base_layer: nn.Module, r=8):
        super().__init__()
        self.r = r
        self.__dict__.update(base_layer.__dict__)
        self.ori_forward = base_layer.forward

    def forward(self, x):
        with torch.no_grad():
            y = self.ori_forward(x)

        x = x.view(-1, self.A.shape[0])
        return y + torch.mm(torch.mm(x, self.A), self.B).view(*y.shape)


class Linear(LoraModule):
    def __init__(self, base_layer: nn.Linear, r=8):
        super().__init__(base_layer, r)
        self.A = nn.Parameter(torch.randn(self.in_features, r))
        self.B = nn.Parameter(torch.zeros(r, self.out_features))


class Conv1D(LoraModule, nn.Conv1d):
    def __init__(self, base_layer: nn.Conv1d, r=8):
        super().__init__(base_layer, r)
        self.A = nn.Parameter(torch.randn(self.in_channels, r))
        self.B = nn.Parameter(torch.zeros(r, self.out_channels))


class Embedding(LoraModule, nn.Embedding):
    def __init__(self, base_layer: nn.Embedding, r=8):
        super().__init__(base_layer, r)
        self.A = nn.Parameter(torch.randn(self.num_embeddings, r))
        self.B = nn.Parameter(torch.zeros(r, self.embedding_dim))


class ModelWarp:
    """
    Usages:
        .. code-block:: python

            model = ...
            model_warp = ModelWarp(include=['to_qkv'])
            model_warp.warp(model)

            # define your train step
            opti = ...
            model(data)
            ...

            # save the additional weight
            state_dict = model_warp.state_dict()
            torch.save(state_dict, save_path)

            # load the additional weight
            model_warp.load_state_dict(state_dict)
            # or load directly by model with warped
            model.load_state_dict(state_dict, strict=False)
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
        self.layers = OrderedDict()

    def warp(self, model: nn.Module):
        model.requires_grad_(False)
        layers = torch_utils.ModuleManager.get_module_by_name(model, include=self.include, exclude=self.exclude)
        if len(layers) == 0:
            warnings.warn(f'can not find any layer by include={self.include} and exclude={self.exclude}')

        for current_m, name, full_name in layers:
            new = self.get_new_module(getattr(current_m, name))
            if new:
                new.to(torch_utils.ModuleInfo.possible_device(current_m))
                setattr(current_m, name, new)
                self.layers[full_name] = new
            else:
                warnings.warn(f'not support lora layer type for {full_name}')

        return model

    def get_new_module(self, module: nn.Module):
        for old, new in self.layer_mapping.items():
            if isinstance(module, old):
                return new(module, r=self.r)

    def state_dict(self, **kwargs):
        state_dict = OrderedDict()
        for full_name, layer in self.layers.items():
            state_dict[full_name + '.A'] = layer.A
            state_dict[full_name + '.B'] = layer.B

        return state_dict

    def load_state_dict(self, state_dict, **kwargs):
        for full_name, layer in self.layers.items():
            layer.A = state_dict[full_name + '.A']
            layer.B = state_dict[full_name + '.B']

