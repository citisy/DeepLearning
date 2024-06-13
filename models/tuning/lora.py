import math
import torch
from torch import nn
from torch.nn import functional as F
import warnings
from utils import torch_utils
from collections import OrderedDict


class LoraModule(nn.Module):
    def __init__(self, base_layer: nn.Module, r=8, multiplier=1.0, alpha=1, drop_prob=0.):
        self.r = r
        self.scale = multiplier * alpha / r

        self.__dict__.update(base_layer.__dict__)
        self.ori_forward = base_layer.forward
        self.dropout = nn.Dropout(drop_prob)

    def initialize_layers(self):
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        with torch.no_grad():
            y = self.ori_forward(x)

        h = self.lora_call(x)
        h = self.dropout(h) * self.scale
        return y + h

    def lora_call(self, x):
        raise NotImplementedError

    def dewarp(self):
        del self.down
        del self.up
        torch.cuda.empty_cache()


class Linear(LoraModule):
    def __init__(self, base_layer: nn.Linear, r=8, **kwargs):
        super().__init__(base_layer, r, **kwargs)
        self.down = torch.nn.Linear(self.in_features, r, bias=False)
        self.up = torch.nn.Linear(r, self.out_features, bias=False)

        self.initialize_layers()

    def lora_call(self, x):
        return self.up(self.down(x))

    def fuse(self):
        self.weight.data += self.up.weight @ self.down.weight * self.scale
        self.forward = self.ori_forward
        self.dewarp()


class Embedding(LoraModule):
    def __init__(self, base_layer: nn.Embedding, r=8, **kwargs):
        super().__init__(base_layer, r, **kwargs)
        self.down = torch.nn.Linear(self.num_embeddings, r, bias=False)
        self.up = torch.nn.Linear(r, self.embedding_dim, bias=False)

        self.initialize_layers()

    def lora_call(self, x):
        emb = (self.up.weight @ self.down.weight).T
        return F.embedding(
            x, emb, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse
        )

    def fuse(self):
        self.weight.data += (self.up.weight @ self.down.weight).T * self.scale
        self.forward = self.ori_forward
        self.dewarp()


class Conv2d(LoraModule):
    def __init__(self, base_layer: nn.Conv2d, r=8, **kwargs):
        super().__init__(base_layer, r, **kwargs)
        self.down = nn.Conv2d(base_layer.in_channels, r, base_layer.kernel_size, base_layer.stride, base_layer.padding, bias=False)
        self.up = nn.Conv2d(r, base_layer.out_channels, 1, bias=False)

        self.initialize_layers()

    def lora_call(self, x):
        return self.up(self.down(x))

    def fuse(self):
        raise "conv layer can not fuse"


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
        nn.Embedding: Embedding,
        nn.Conv2d: Conv2d
    }

    def __init__(self, include=(), exclude=(), r=8):
        self.include = include
        self.exclude = exclude
        self.r = r
        self.layers = []
        self.model = None

    def warp(self, model: nn.Module):
        model.requires_grad_(False)
        layers = torch_utils.ModuleManager.get_module_by_key(model, include=self.include, exclude=self.exclude)
        if len(layers) == 0:
            warnings.warn(f'can not find any layer by include={self.include} and exclude={self.exclude}')

        for current_m, name, full_name in layers:
            new = self.get_new_module(getattr(current_m, name))
            if new:
                new.to(torch_utils.ModuleInfo.possible_device(current_m))
                setattr(current_m, name, new)
                self.layers.append(full_name)
            else:
                warnings.warn(f'not support lora layer type for {full_name}')

        self.model = model
        return model

    def get_new_module(self, module: nn.Module):
        for old, new in self.layer_mapping.items():
            if isinstance(module, old):
                return new(module, r=self.r)

    def state_dict(self, **kwargs):
        state_dict = OrderedDict()
        for full_name in self.layers:
            # note, to avoid the layer is changed
            layer = torch_utils.ModuleManager.get_module_by_name(self.model, full_name)
            state_dict[full_name + '.A'] = layer.A
            state_dict[full_name + '.B'] = layer.B

        return state_dict

    def load_state_dict(self, state_dict, **kwargs):
        for full_name in self.layers:
            layer = torch_utils.ModuleManager.get_module_by_name(self.model, full_name)
            layer.A = state_dict[full_name + '.A']
            layer.B = state_dict[full_name + '.B']

    def fuse(self):
        for full_name in self.layers:
            layer = torch_utils.ModuleManager.get_module_by_name(self.model, full_name)
            if hasattr(layer, 'fuse'):
                layer.fuse()

    def dewarp(self):
        for full_name in self.layers:
            layer = torch_utils.ModuleManager.get_module_by_name(self.model, full_name)
            if hasattr(layer, 'fuse'):
                layer.dewarp()
