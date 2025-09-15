import torch
from torch import nn

from .. import bundles
from . import bert
from ..layers import Linear
from utils import torch_utils


class Config(bundles.Config):
    default_model = ''

    backbone = dict(
        hidden_size=1024,
        vocab_size=250002,
        pad_id=1,
        num_attention_heads=16,
        num_hidden_layers=24,
        n_segment=1,
        max_seq_len=8194,
    )

    @classmethod
    def make_full_config(cls) -> dict:
        config_dict = {
            '': cls.backbone
        }
        return config_dict


class WeightConverter:
    convert_dict = {
        'embeddings.word_embeddings': 'backbone.embedding.token',
        'embeddings.position_embeddings': 'backbone.embedding.position',
        'embeddings.token_type_embeddings': 'backbone.embedding.segment',
        'embeddings.LayerNorm': 'backbone.embedding.head.0',

        'encoder.layer.{0}.attention.self.query': 'backbone.encoder.{0}.attn_res.fn.to_qkv.0',
        'encoder.layer.{0}.attention.self.key': 'backbone.encoder.{0}.attn_res.fn.to_qkv.1',
        'encoder.layer.{0}.attention.self.value': 'backbone.encoder.{0}.attn_res.fn.to_qkv.2',
        'encoder.layer.{0}.attention.output.dense': 'backbone.encoder.{0}.attn_res.fn.to_out.linear',
        'encoder.layer.{0}.attention.output.LayerNorm': 'backbone.encoder.{0}.attn_res.norm',
        'encoder.layer.{0}.intermediate.dense': 'backbone.encoder.{0}.ff_res.fn.0.linear',
        'encoder.layer.{0}.output.dense': 'backbone.encoder.{0}.ff_res.fn.1.linear',
        'encoder.layer.{0}.output.LayerNorm': 'backbone.encoder.{0}.ff_res.norm',

        'pooler.dense': 'head.fcn.linear',
    }

    @classmethod
    def from_hf(cls, state_dict):
        """convert weights from huggingface model to my own model

        Usage:
            .. code-block:: python

                state_dict = WeightLoader.from_hf(...)
                state_dict = WeightConverter.from_hf(state_dict)
                Model(...).load_state_dict(state_dict)

        """
        state_dict = torch_utils.Converter.convert_keys(state_dict, cls.convert_dict)
        return state_dict


class Model(nn.Module):
    """refer to `transformers.XLMRobertaModel`"""
    def __init__(self, backbone_config=Config.backbone):
        super().__init__()
        self.backbone = bert.Bert(**backbone_config)
        self.head = Head(self.backbone.out_features)

    def forward(self, input_ids, attention_mask):
        if self.training:
            raise NotImplementedError
        x = self.backbone(input_ids, attention_mask=attention_mask)
        x = self.head(x)
        return x


class Head(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fcn = Linear(hidden_size, hidden_size, mode='la', act=nn.Tanh())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.fcn(first_token_tensor)
        return pooled_output
