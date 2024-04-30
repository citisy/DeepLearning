import warnings
import torch
from torch import nn
from utils import torch_utils
from functools import partial
from .. import attentions
from collections import OrderedDict


class PrefixEncoder(nn.Module):
    def __init__(
            self, n_heads, head_dim, num_hidden_layers,
            pre_seq_len=4, prefix_projection=False, prefix_hidden_size=None,
            drop_prob=0.1
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.pre_seq_len = pre_seq_len
        model_dim = n_heads * head_dim
        self.prefix_projection = prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = nn.Sequential(
                nn.Embedding(pre_seq_len, model_dim),
                nn.Linear(model_dim, prefix_hidden_size),
                nn.Tanh(),
                nn.Linear(prefix_hidden_size, num_hidden_layers * 2 * model_dim)
            )
        else:
            self.embedding = nn.Embedding(pre_seq_len, num_hidden_layers * 2 * model_dim)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, prefix: torch.Tensor):
        past_key_values = self.embedding(prefix)

        # todo: support linear attention
        past_key_values = past_key_values.view(
            prefix.shape[0],
            self.pre_seq_len,
            self.num_hidden_layers * 2,
            self.n_heads,
            self.head_dim
        )
        past_key_values = self.dropout(past_key_values)
        # (b, s, 2N, n, d) -> (2N, b, n, s, d)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


class ModelWarp:
    """
    Usages:
        .. code-block:: python

            # model having an attention layer with name of 'attend'
            model = ...
            model_warp = ModelWarp(n_heads, head_dim, include=['attend'])
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
    """

    def __init__(
            self, n_heads, head_dim,
            include=(), exclude=(),
            **prefix_encoder_kwargs
    ):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.include = include
        self.exclude = exclude
        self.layers = OrderedDict()
        self.attn_layers = OrderedDict()
        self.prefix_encoder_kwargs = prefix_encoder_kwargs
        self.pre_seq_len = -1

    def warp(self, model: nn.Module):
        torch_utils.ModuleManager.freeze_module(model, allow_train=True)
        # add prefix layers
        layers = torch_utils.ModuleManager.get_module_by_name(model, include=self.include, exclude=self.exclude, is_last=True)
        if len(layers) == 0:
            warnings.warn(f'can not find any layer by include={self.include} and exclude={self.exclude}')

        for current_m, name, full_name in layers:
            new = attentions.LearnedMemoryAttend(getattr(current_m, name))
            new.to(torch_utils.ModuleInfo.possible_device(current_m))
            setattr(current_m, name, new)
            self.attn_layers[full_name] = new

        # add prefix_encoder layer
        prefix_encoder = PrefixEncoder(
            self.n_heads,
            self.head_dim,
            len(self.layers),
            **self.prefix_encoder_kwargs
        )
        prefix_encoder.to(torch_utils.ModuleInfo.possible_device(model))
        self.layers['prefix_encoder'] = prefix_encoder

        model.prefix_encoder = prefix_encoder
        self.pre_seq_len = model.prefix_encoder.pre_seq_len
        model.forward = partial(self.model_forward, model, model.forward)
        return model

    def model_forward(
            self, base_layer, base_forward, x, *args,
            attention_mask=None,
            **kwargs
    ):
        b = x.shape[0]
        prefix_tokens = torch.arange(self.pre_seq_len).long().to(x.device)
        prefix_tokens = prefix_tokens.unsqueeze(0).expand(b, -1)
        past_key_values = base_layer.prefix_encoder(prefix_tokens)

        per_block_kwargs = []
        for i in range(len(self.attn_layers)):
            per_block_kwargs.append(dict(
                mem_kv=past_key_values[i]
            ))

        if attention_mask is not None:
            attention_mask = torch.cat((
                torch.ones((x.shape[0], self.pre_seq_len)).to(attention_mask),
                attention_mask
            ), dim=1)

        return base_forward(
            x, *args,
            attention_mask=attention_mask,
            per_block_kwargs=per_block_kwargs,
            **kwargs
        )

    def state_dict(self, **kwargs):
        state_dict = OrderedDict()
        for full_name, layer in self.layers.items():
            state_dict[full_name] = layer.state_dict()

        return state_dict

    def load_state_dict(self, state_dict, **kwargs):
        for full_name, layer in self.layers.items():
            layer.load_state_dict(state_dict[full_name])