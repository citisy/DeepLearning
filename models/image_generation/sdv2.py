import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn, einsum
from utils import torch_utils
from . import ldm, ddpm, VAE, sdv1


class Config(sdv1.Config):
    """only for inference"""

    # for OpenCLIPEmbedder layer output
    LAST = 'last'
    PENULTIMATE = 'penultimate'
    POOLED = 'pooled'

    v2_model = dict(
        scale=9.,
        objective=ddpm.Config.PRED_Z,
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
                backbone_config=cls.v2_backbone,
                vae_config=cls.vae
            ),

            # support version v2-v, v2-768, v2.1, v2.1-768
            v2_v=dict(
                model_config=cls.v2_v_model,
                backbone_config=cls.v2_backbone,
                vae_config=cls.vae
            ),

            v2_inpaint=dict(
                model_config=cls.v2_inpaint_model,
                backbone_config=cls.v2_backbone,
                vae_config=cls.v2_inpaint_vae
            ),

            v2_midas=dict(
                model_config=cls.v2_inpaint_model,  # same to inpaint
                backbone_config=cls.v2_backbone,
                vae_config=cls.v2_midas_vae
            ),

            x4=dict(
                model_config=cls.x4_model,
                backbone_config=cls.x4_backbone,
                vae_config=cls.x4_vae
            ),

            v2_unclip_l=dict(
                model_config=cls.v2_unclip_model,
                backbone_config=cls.v2_unclip_l_backbone,
                vae_config=cls.vae
            ),

            v2_unclip_h=dict(
                model_config=cls.v2_unclip_model,
                backbone_config=cls.v2_unclip_h_backbone,
                vae_config=cls.vae
            )

        )
        return config_dict


class Model(ldm.Model):
    def make_cond(self, cond_config=dict(), **kwargs):
        return OpenCLIPEmbedder(**cond_config)


class OpenCLIPEmbedder(nn.Module):
    """Uses the OpenCLIP transformer encoder for text
    see https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K
    """

    def __init__(self, arch="ViT-H-14", pretrain_model=None, layer=Config.PENULTIMATE,
                 return_pooled=True, legacy=True):
        super().__init__()
        import open_clip  # pip install open-clip-torch>=2.20.0

        # if arch="ViT-H-14", use pretrain_model = 'laion2b_s32b_b79k'
        # if arch="ViT-bigG-14", use pretrain_model = 'laion2b_s39b_b160k'
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrain_model)
        del model.visual
        self.model = model
        self.tokenizer = open_clip.tokenize

        self.max_length = model.context_length  # 77
        self.output_size = model.transformer.width  # 1024
        self.return_pooled = return_pooled
        self.legacy = legacy
        self.layer = layer

    def forward(self, text):
        device = self.model.attn_mask.device
        tokens = self.tokenizer(text).to(device)
        return self.encode_with_transformer(tokens)

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        if self.legacy:
            x = x[self.layer]
            x = self.model.ln_final(x)
            return x
        else:
            o = x["last"]
            o = self.model.ln_final(o)

            if self.layer == Config.POOLED or self.return_pooled:
                pooled = self.pool(o, text)
                x["pooled"] = pooled
                if self.return_pooled:
                    return x[self.layer], x['pooled']

            return x[self.layer]

    def pool(self, x, text):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        idx = torch.arange(x.shape[0]), text.argmax(dim=-1)
        x = x[idx] @ self.model.text_projection
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        outputs = {}
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:  # the last block, for penultimate
                outputs["penultimate"] = x.permute(1, 0, 2)  # LND -> NLD
                # break

            if self.model.transformer.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)

        outputs["last"] = x.permute(1, 0, 2)  # LND -> NLD
        return outputs
