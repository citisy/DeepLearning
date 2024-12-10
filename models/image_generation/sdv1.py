import torch
from torch import nn, einsum
from . import ldm, k_diffusion
from .ldm import WeightLoader
from ..multimodal_pretrain import CLIP


class Config(ldm.Config):
    """only for inference"""
    # support version v1, v1.*

    # for CLIPEmbedder layer output
    LAST = 'last'
    RAW_HIDDEN = 'raw_hidden'
    NORM_HIDDEN = 'norm_hidden'
    POOLED = 'pooled'

    cond = dict(
        is_proj=False,
        **CLIP.Config.openai_text_large,
        layer=LAST,
    )

    v1_5sampler = dict(
        name=ldm.Config.EULER,
        **k_diffusion.Config.get(''),
    )

    default_model = 'v1'

    @classmethod
    def make_full_config(cls):
        config_dict = {
            'v1': dict(
                model_config=cls.model,
                sampler_config=cls.sampler,
                backbone_config=cls.backbone,
                vae_config=cls.vae,
                cond_config=cls.cond,
            ),

            'v1.5': dict(
                model_config=cls.model,
                sampler_config=cls.v1_5sampler,
                backbone_config=cls.backbone,
                vae_config=cls.vae,
                cond_config=cls.cond,
            )
        }
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
    def __init__(self, layer=Config.RAW_HIDDEN, layer_idx=None, return_pooled=False, **kwargs):
        super().__init__()
        self.transformer = CLIP.TextModel(**kwargs)

        self.max_length = kwargs['max_seq_len']
        self.output_size = kwargs['output_size']
        self.layer = layer
        self.layer_idx = layer_idx
        self.return_pooled = return_pooled
        self.callback = Callback(layer_idx)

    def forward(self, text_ids, **kwargs):
        x = self.transformer.text_model.backbone(text_ids, callback_fn=self.callback)
        pooled_output = None

        if self.layer == Config.RAW_HIDDEN:  # without norm
            z = self.callback.cache_hidden_state

        elif self.layer == Config.NORM_HIDDEN:  # with norm
            z = self.callback.cache_hidden_state
            z = self.transformer.text_model.neck(z)

        elif self.layer == Config.LAST:
            z = self.transformer.text_model.neck(x)

        elif self.layer == Config.POOLED:
            h = self.transformer.text_model.neck(x)
            pooled_output = self.transformer.text_model.head(text_ids, h)
            z = pooled_output[:, None, :]

        else:
            raise f'do not support layer={self.layer}'

        if self.return_pooled:
            if self.layer in {Config.RAW_HIDDEN, Config.NORM_HIDDEN}:
                h = self.transformer.text_model.neck(x)
                pooled_output = self.transformer.text_model.head(text_ids, h)

            elif self.layer == Config.LAST:
                pooled_output = self.transformer.text_model.head(text_ids, z)

            elif self.layer == Config.POOLED:
                pass

            else:
                raise f'return_pooled do not support layer={self.layer}'

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
