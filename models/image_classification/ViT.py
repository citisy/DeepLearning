import torch
from torch import nn

from utils import torch_utils
from . import BaseImgClsModel, make_backbone_fn
from .. import bundles
from ..text_pretrain.transformers import TransformerSequential
from .. import attentions, embeddings


class Config(bundles.Config):
    backbone_B_16 = dict(
        in_ch=3,
        input_size=224,
        embed_dim=768,
        patch_size=16,
        ff_hidden_size=3072,
        depth=12,
        heads=8,
        dropout=0.1,
        emb_dropout=0.1
    )

    backbone_B_16_H_12 = dict(
        in_ch=3,
        input_size=224,
        embed_dim=768,
        patch_size=16,
        ff_hidden_size=3072,
        depth=12,
        heads=12,
        dropout=0.1,
        emb_dropout=0.1
    )

    default_model = 'B_16'

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            'B_16': dict(
                backbone_config=cls.backbone_B_16
            ),
            'B_16_H_12': dict(
                backbone_config=cls.backbone_B_16_H_12
            )
        }


class WeightConverter:
    @classmethod
    def from_official_flax(cls, state_dict):
        """https://console.cloud.google.com/storage/browser/vit_models/imagenet21k"""
        convert_dict = {

        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
        return state_dict

    @classmethod
    def from_modelscope(cls, state_dict):
        """https://modelscope.cn/models/iic/cv_vit-base_image-classification_Dailylife-labels/summary"""
        convert_dict = {
            'backbone.cls_token': 'backbone.embedding.cls',
            'backbone.pos_embed': 'backbone.embedding.position.weight',
            'backbone.patch_embed.projection': 'backbone.embedding.patch.fn.0',
            'backbone.layers.{0}.attn.qkv': 'backbone.encoder.{0}.attn_res.fn.to_qkv',
            'backbone.layers.{0}.attn.proj': 'backbone.encoder.{0}.attn_res.fn.to_out.linear',
            'backbone.layers.{0}.ln1': 'backbone.encoder.{0}.attn_res.norm',
            'backbone.layers.{0}.ln2': 'backbone.encoder.{0}.ff_res.norm',
            'backbone.layers.{0}.ffn.layers.0.0': 'backbone.encoder.{0}.ff_res.fn.0.linear',
            'backbone.layers.{0}.ffn.layers.1': 'backbone.encoder.{0}.ff_res.fn.1.linear',

            'backbone.ln1': 'backbone.final_norm',
            'head.layers.head': 'head.0'
        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)

        state_dict['backbone.embedding.cls'] = state_dict['backbone.embedding.cls'][0, 0]
        state_dict['backbone.embedding.position.weight'] = state_dict['backbone.embedding.position.weight'][0]
        return state_dict


class Model(BaseImgClsModel):
    """refer to
    paper:
        - [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929.pdf)
    code:
        - https://github.com/google-research/vision_transformer
        - https://github.com/lucidrains/vit-pytorch

    """

    def __init__(
            self,
            out_features, in_ch=3, input_size=256,
            out_module=None, backbone_config=Config.backbone_B_16, **kwargs
    ):
        backbone_config.setdefault('in_ch', in_ch)
        backbone_config.setdefault('input_size', input_size)

        backbone = Backbone(**backbone_config)
        neck = ClsNeck()
        head = out_module or nn.Sequential(
            # nn.Linear(backbone.embed_dim, backbone.embed_dim),
            nn.Linear(backbone.embed_dim, out_features)
        )

        super().__init__(
            in_module=nn.Identity(),  # placeholder
            backbone=backbone,
            neck=neck,
            head=head,
            **kwargs
        )


@make_backbone_fn.add_register('ViT')
class Backbone(nn.Module):
    def __init__(self, in_ch=3, input_size=256, embed_dim=1024,
                 patch_size=32, ff_hidden_size=2048, depth=6, heads=8, drop_prob=0.1,
                 **kwargs):
        super().__init__()

        self.in_ch = in_ch
        self.input_size = input_size
        self.embed_dim = embed_dim

        self.embedding = VisionEmbedding(embed_dim, input_size, patch_size)
        self.dropout = nn.Dropout(drop_prob)
        self.encoder = TransformerSequential(
            embed_dim, heads, ff_hidden_size,

            attention_fn=attentions.CrossAttention2D,

            fn_kwargs=dict(
                use_conv=False,
                separate=False
            ),
            ff_kwargs=dict(
                act=nn.GELU()
            ),

            norm_first=True,
            num_blocks=depth
        )
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.encoder(x)
        x = self.final_norm(x)
        return x


class VisionEmbedding(nn.Module):
    def __init__(self, embed_dim, image_size, patch_size, patch_bias=True):
        super().__init__()
        self.patch = embeddings.PatchEmbedding(embed_dim, patch_size, bias=patch_bias)
        self.cls = nn.Parameter(torch.randn(embed_dim))

        # note, add 1 is in order to apply the class_embedding
        num_positions = (image_size // patch_size) ** 2 + 1
        self.position = embeddings.LearnedPositionEmbedding3D(num_positions, embed_dim)

    def forward(self, x):
        patch_embeds = self.patch(x)

        class_embeds = self.cls.expand(x.shape[0], 1, -1)
        x = torch.cat([class_embeds, patch_embeds], dim=1)

        x = x + self.position(x)
        return x


class ClsNeck(nn.Module):
    """for class prediction"""

    def forward(self, x):
        x = x[:, 0, :]
        return x
