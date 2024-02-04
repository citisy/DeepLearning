import open_clip
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn, einsum
from utils import torch_utils
from . import ldm, ddpm, VAE, sdv1
from .ldm import convert_weights


class Config(ldm.Config):
    """only for inference"""
    POOLED = 0
    LAST = 1
    PENULTIMATE = 2

    model = dict(
        scale=9.,
        objective=ddpm.Config.PRED_Z,
    )

    backbone = dict(
        head_dim=64,
        use_linear_in_transformer=True
    )

    v_model = dict(
        scale=9.,
        objective=ddpm.Config.PRED_V,
    )

    inpaint_model = dict(
        objective=ddpm.Config.PRED_V,
        conditioning_key=ldm.Config.HYBRID
    )

    inpaint_vae = dict(
        z_ch=9,
        ch_mult=(1, 2, 4, 4),
        attn_layers=[]
    )

    midas_vae = dict(
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

    unclip_model = dict(
        scale=9.,
        objective=ddpm.Config.PRED_V,
        conditioning_key=ldm.Config.CROSSATTN_ADM
    )

    unclip_vae = dict(
        attn_type=VAE.Config.VANILLA_XFORMERS,
        **VAE.Config.backbone_32x32x4
    )

    unclip_l_backbone = dict(
        num_classes="sequential",
        adm_in_channels=1536,
        **backbone
    )

    unclip_h_backbone = dict(
        num_classes="sequential",
        adm_in_channels=2048,
        **backbone
    )

    @classmethod
    def get(cls, name=None):
        config_dict = dict(
            vanilla=dict(  # support version v2
                model_config=cls.model,
                backbone_config=cls.backbone,
                vae_config=cls.vae
            ),

            v=dict(  # support version v2-v, v2-768, v2.1, v2.1-768
                model_config=cls.v_model,
                backbone_config=cls.backbone,
                vae_config=cls.vae
            ),

            inpaint=dict(
                model_config=cls.inpaint_model,
                backbone_config=cls.backbone,
                vae_config=cls.inpaint_vae
            ),

            midas=dict(
                model_config=cls.inpaint_model,  # same to inpaint
                backbone_config=cls.backbone,
                vae_config=cls.midas_vae
            ),

            x4=dict(
                model_config=cls.x4_model,
                backbone_config=cls.x4_backbone,
                vae_config=cls.x4_vae
            ),

            unclip_l=dict(
                model_config=cls.unclip_model,
                backbone_config=cls.unclip_l_backbone,
                vae_config=cls.unclip_vae
            ),

            unclip_h=dict(
                model_config=cls.unclip_model,
                backbone_config=cls.unclip_h_backbone,
                vae_config=cls.unclip_vae
            )

        )
        return config_dict.get(name, 'vanilla')


class Model(ldm.Model):
    def make_cond(self, cond_config=dict(), **kwargs):
        return FrozenOpenCLIPEmbedder(**cond_config)


class FrozenOpenCLIPEmbedder(nn.Module):
    """Uses the OpenCLIP transformer encoder for text
    see https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    """

    def __init__(self, pretrain_model=None, arch="ViT-H-14",
                 max_length=77, layer=Config.PENULTIMATE):
        super().__init__()

        # pretrain_model = 'laion2b_s32b_b79k'
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrain_model)
        del model.visual
        self.model = model

        self.max_length = max_length
        self.output_size = 1024
        self.layer = layer
        if self.layer == Config.LAST:
            self.layer_idx = 0
        elif self.layer == Config.PENULTIMATE:
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def forward(self, text):
        device = self.model.attn_mask.device
        tokens = open_clip.tokenize(text).to(device)
        z = self.encode_with_transformer(tokens)
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x
