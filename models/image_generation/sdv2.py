import open_clip
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn, einsum
from utils import torch_utils
from . import ldm, ddpm, VAE
from .ldm import convert_weights


class Config(ldm.Config):
    POOLED = 0
    LAST = 1
    PENULTIMATE = 2

    in_module = dict(
        layer=PENULTIMATE
    )

    backbone = dict(
        context_dim=1024,
        head_dim=64
    )

    v_model = dict(
        objective=ddpm.Config.PRED_V
    )

    inpaint_model = dict(
        objective=ddpm.Config.PRED_V,
        conditioning_key=ldm.Config.HYBRID
    )

    inpaint_backbone = dict(
        in_ch=9,
        **backbone
    )

    midas_backbone = dict(
        in_ch=5,
        **backbone
    )

    x4_model = dict(
        objective=ddpm.Config.PRED_V,
        conditioning_key=ldm.Config.HYBRID_ADM,
        scale_factor=0.08333
    )

    x4_backbone = dict(
        in_ch=7,
        unit_dim=256,
        attend_layers=(1, 2, 3),
        ch_mult=(1, 2, 2, 4),
        num_heads=8,
        context_dim=1024,
    )

    x4_head = dict(
        img_ch=3,
        z_ch=4,
        ch_mult=(1, 2, 4),
        attn_layers=[]
    )

    unclip_model = dict(
        objective=ddpm.Config.PRED_V,
        conditioning_key=ldm.Config.CROSSATTN_ADM
    )

    unclip_head = dict(
        img_ch=3,
        backbone_config=dict(
            attn_type=VAE.Config.VANILLA_XFORMERS,
            **VAE.Config.backbone_32x32x4
        ),
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
            vanilla=dict(
                model_config=cls.model,
                in_module_config=cls.in_module,
                backbone_config=cls.backbone,
                head_config=cls.head
            ),

            v=dict(
                model_config=cls.v_model,
                in_module_config=cls.in_module,
                backbone_config=cls.backbone,
                head_config=cls.head
            ),

            inpaint=dict(
                model_config=cls.inpaint_model,
                in_module_config=cls.in_module,
                backbone_config=cls.inpaint_backbone,
                head_config=cls.head
            ),

            midas=dict(
                model_config=cls.inpaint_model,  # same to inpaint
                in_module_config=cls.in_module,
                backbone_config=cls.midas_backbone,
                head_config=cls.head
            ),

            x4=dict(
                model_config=cls.x4_model,
                in_module_config=cls.in_module,
                backbone_config=cls.x4_backbone,
                head_config=cls.x4_head
            ),

            unclip_l=dict(
                model_config=cls.unclip_model,
                in_module_config=cls.in_module,
                backbone_config=cls.unclip_l_backbone,
                head_config=cls.head
            ),

            unclip_h=dict(
                model_config=cls.unclip_model,
                in_module_config=cls.in_module,
                backbone_config=cls.unclip_h_backbone,
                head_config=cls.head
            )

        )
        return config_dict.get(name, 'vanilla')


class Model(ldm.Model):
    def __init__(self, *args, in_module_config=Config.in_module, **kwargs):
        in_module = FrozenOpenCLIPEmbedder(**in_module_config)

        super().__init__(
            *args,
            in_module=in_module,
            **kwargs
        )


class FrozenOpenCLIPEmbedder(nn.Module):
    """Uses the OpenCLIP transformer encoder for text
    see https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    """

    def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k",
                 max_length=77, layer=Config.PENULTIMATE):
        super().__init__()

        model, _, _ = open_clip.create_model_and_transforms(arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.max_length = max_length
        self.layer = layer
        if self.layer == Config.LAST:
            self.layer_idx = 0
        elif self.layer == Config.PENULTIMATE:
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
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
