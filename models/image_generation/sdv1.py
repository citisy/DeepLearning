import torch
from torch import nn, einsum
from . import ldm
from .ldm import WeightLoader
from ..text_image_pretrain import CLIP


class Config(ldm.Config):
    """only for inference"""
    # support version v1, v1.*

    # for CLIPEmbedder layer output
    LAST = 'last'
    HIDDEN = 'hidden'
    POOLED = 'pooled'
    PENULTIMATE = 'penultimate'

    cond = dict(
        is_proj=False,
        **CLIP.Config.openai_text_large,
        layer=LAST,
    )

    default_model = 'v1'

    @classmethod
    def make_full_config(cls):
        config_dict = dict(
            v1=dict(
                model_config=cls.model,
                backbone_config=cls.backbone,
                vae_config=cls.vae,
                cond_config=cls.cond,
            )
        )
        return config_dict


class WeightConverter(ldm.WeightConverter):
    cond_convert_dict = {
        'cond_stage_model.transformer.' + k: 'cond.transformer.' + v
        for k, v in CLIP.WeightConverter.openai_convert_dict.items()
    }


class Model(ldm.Model):
    """
    https://github.com/CompVis/stable-diffusion
    """

    def make_cond(self, cond_config=dict(), **kwargs):
        return CLIPEmbedder(**cond_config)


class CLIPEmbedder(nn.Module):
    def __init__(self, layer=Config.HIDDEN, layer_idx=None, return_pooled=False, **kwargs):
        super().__init__()
        self.transformer = CLIP.TextModel(**kwargs)

        self.max_length = kwargs['max_seq_len']  # 77
        self.output_size = kwargs['output_size']  # 1024
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = return_pooled
        self.callback = Callback(layer_idx)

    def forward(self, text_ids, **kwargs):
        x = self.transformer.text_model.backbone(text_ids, callback_fn=self.callback)
        pooled_output = None

        if self.layer == Config.HIDDEN:
            z = self.callback.cache_hidden_state
        elif self.layer == Config.PENULTIMATE:
            z = self.callback.cache_hidden_state
            z = self.transformer.text_model.neck(z)
        elif self.layer == Config.LAST:
            z = self.transformer.text_model.neck(x)
        elif self.layer == Config.POOLED:
            h = self.transformer.text_model.neck(x)
            pooled_output = self.transformer.text_model.head(text_ids, h)
            z = pooled_output[:, None, :]
        else:
            raise f'Do not support layer={self.layer}'

        if self.return_pooled:
            if self.layer == Config.LAST:
                pooled_output = self.transformer.text_model.head(text_ids, z)
            elif self.layer == Config.HIDDEN:
                h = self.transformer.text_model.neck(x)
                pooled_output = self.transformer.text_model.head(text_ids, h)

            return z, pooled_output

        else:
            return z


class Callback(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.cache_hidden_state = None

    def forward(self, i, h):
        if i == self.layer_idx:
            self.cache_hidden_state = h
