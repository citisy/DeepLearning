import torch
from torch import nn

from utils import torch_utils
from . import BaseImgClsModel
from .. import bundles
from ..attentions import CrossAttention3D
from ..embeddings import PatchEmbedding, LearnedPositionEmbedding
from ..text_pretrain.transformers import TransformerSequential

default_config = dict(
    in_ch=3,
    input_size=256,
    embed_dim=1024,
    patch_size=32,
    ff_hidden_size=2048,
    depth=6,
    heads=8,
    dropout=0.1,
    emb_dropout=0.1
)


class Config(bundles.Config):
    @classmethod
    def make_full_config(cls) -> dict:
        return {
            'B_16': dict(
                in_ch=3,
                input_size=224,
                embed_dim=768,
                patch_size=16,
                ff_hidden_size=3072,
                depth=12,
                heads=8,
                dropout=0.1,
                emb_dropout=0.1
            ),
        }


class WeightConverter:
    @classmethod
    def from_official_flax(cls, state_dict):
        """https://console.cloud.google.com/storage/browser/vit_models/imagenet21k"""
        convert_dict = {

        }

        state_dict = torch_utils.Converter.convert_keys(state_dict, convert_dict)
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
            out_module=None, backbone_config=default_config, **kwargs
    ):
        backbone_config.setdefault('in_ch', in_ch)
        backbone_config.setdefault('input_size', input_size)

        backbone = Backbone(**backbone_config)
        neck = ClsNeck()
        head = out_module or nn.Sequential(
            nn.Linear(backbone.embed_dim, backbone.embed_dim),
            nn.Linear(backbone.embed_dim, out_features)
        )

        super().__init__(
            in_module=nn.Identity(),  # placeholder
            backbone=backbone,
            neck=neck,
            head=head,
            **kwargs
        )


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

            attention_fn=CrossAttention3D,

            fn_kwargs=dict(
                use_conv=False
            ),
            ff_kwargs=dict(
                act=nn.GELU()
            ),

            norm_first=True,
            num_blocks=depth
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        x = self.encoder(x)
        return x


class VisionEmbedding(nn.Module):
    def __init__(self, embed_dim, image_size, patch_size):
        super().__init__()
        self.patch = PatchEmbedding(embed_dim, patch_size, bias=True)
        self.cls = nn.Parameter(torch.randn(embed_dim))

        # note, add 1 is in order to apply the class_embedding
        num_positions = (image_size // patch_size) ** 2 + 1
        self.position = LearnedPositionEmbedding(num_positions, embed_dim)

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
