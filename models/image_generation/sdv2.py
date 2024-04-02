import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn, einsum
from utils import torch_utils
from . import ldm, ddpm, VAE, sdv1
from .sdv1 import WeightLoader, Model
from ..text_image_pretrain import CLIP


class Config(sdv1.Config):
    """only for inference"""

    v2_model = dict(
        scale=9.,
        objective=ddpm.Config.PRED_Z,
    )

    v2_cond = dict(
        **CLIP.Config.laion_text_H_14,
        layer=sdv1.Config.PENULTIMATE,
        layer_idx=CLIP.Config.laion_text_H_14['num_hidden_layers'] - 2  # second to last state
    )

    v2_backbone = dict(
        head_dim=64,
        use_linear_in_transformer=True
    )

    v2_v_model = dict(
        scale=9.,
        objective=ddpm.Config.PRED_V,
    )

    v2_inpaint_model = dict(
        objective=ddpm.Config.PRED_V,
        conditioning_key=ldm.Config.HYBRID
    )

    v2_inpaint_vae = dict(
        z_ch=9,
        ch_mult=(1, 2, 4, 4),
        attn_layers=[]
    )

    v2_midas_vae = dict(
        z_ch=5,
        ch_mult=(1, 2, 4, 4),
        attn_layers=[]
    )

    x4_model = dict(
        objective=ddpm.Config.PRED_V,
        conditioning_key=ldm.Config.HYBRID_ADM,
        scale_factor=0.08333
    )

    x4_backbone = dict(
        unit_dim=256,
        attend_layers=(1, 2, 3),
        ch_mult=(1, 2, 2, 4),
        num_heads=8,
        context_dim=1024,
    )

    x4_vae = dict(
        z_ch=7,
        ch_mult=(1, 2, 4),
        attn_layers=[]
    )

    v2_unclip_model = dict(
        scale=9.,
        objective=ddpm.Config.PRED_V,
        conditioning_key=ldm.Config.CROSSATTN_ADM
    )

    v2_unclip_l_backbone = dict(
        num_classes=ldm.Config.SEQUENTIAL,
        adm_in_channels=1536,
        **v2_backbone
    )

    v2_unclip_h_backbone = dict(
        num_classes=ldm.Config.SEQUENTIAL,
        adm_in_channels=2048,
        **v2_backbone
    )

    default_model = 'v2'

    @classmethod
    def make_full_config(cls):
        config_dict = dict(
            # support version v2
            v2=dict(
                model_config=cls.v2_model,
                cond_config=cls.v2_cond,
                backbone_config=cls.v2_backbone,
                vae_config=cls.vae
            ),

            # support version v2-v, v2-768, v2.1, v2.1-768
            v2_v=dict(
                model_config=cls.v2_v_model,
                cond_config=cls.v2_cond,
                backbone_config=cls.v2_backbone,
                vae_config=cls.vae
            ),

            v2_inpaint=dict(
                model_config=cls.v2_inpaint_model,
                cond_config=cls.v2_cond,
                backbone_config=cls.v2_backbone,
                vae_config=cls.v2_inpaint_vae
            ),

            v2_midas=dict(
                model_config=cls.v2_inpaint_model,  # same to inpaint
                cond_config=cls.v2_cond,
                backbone_config=cls.v2_backbone,
                vae_config=cls.v2_midas_vae
            ),

            x4=dict(
                model_config=cls.x4_model,
                cond_config=cls.v2_cond,
                backbone_config=cls.x4_backbone,
                vae_config=cls.x4_vae
            ),

            v2_unclip_l=dict(
                model_config=cls.v2_unclip_model,
                cond_config=cls.v2_cond,
                backbone_config=cls.v2_unclip_l_backbone,
                vae_config=cls.vae
            ),

            v2_unclip_h=dict(
                model_config=cls.v2_unclip_model,
                cond_config=cls.v2_cond,
                backbone_config=cls.v2_unclip_h_backbone,
                vae_config=cls.vae
            )

        )
        return config_dict


class WeightConverter(ldm.WeightConverter):
    cond_convert_dict = {
        'cond_stage_model.model.' + k: 'cond.transformer.' + v
        for k, v in CLIP.WeightConverter.laion_convert_dict.items()
    }

    transpose_keys = ('cond.transformer.text_model.proj.weight',)
