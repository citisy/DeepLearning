import torch
from torch import nn
from . import BaseImgClsModel
from ..layers import OutModule, Linear
from ..embeddings import PatchEmbedding, LearnedPositionEmbedding
from ..text_pretrain.transformers import TransformerSequential

default_config = dict(
    in_ch=3,
    input_size=256,
    output_size=1024,
    patch_size=32,
    hidden_size=2048,
    depth=6,
    heads=8,
    dropout=0.1,
    emb_dropout=0.1
)


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
            in_ch=None, input_size=None, out_features=None,
            out_module=None,
            backbone_config=default_config, **kwargs
    ):
        if in_ch:
            backbone_config.update(in_ch=in_ch)
        if input_size:
            backbone_config.update(input_size=input_size)

        backbone = Backbone(**backbone_config)
        neck = ClsNeck(backbone.output_size)
        head = out_module or Linear(backbone.output_size, out_features, mode='nl', norm=nn.LayerNorm(backbone.output_size))

        super().__init__(
            in_module=nn.Identity(),  # placeholder
            backbone=backbone,
            neck=neck,
            head=head,
            **kwargs
        )


class Backbone(nn.Module):
    def __init__(self, in_ch=3, input_size=256, output_size=1024,
                 patch_size=32, hidden_size=2048, depth=6, heads=8, drop_prob=0.1,
                 **kwargs):
        super().__init__()

        self.in_ch = in_ch
        self.input_size = input_size
        self.output_size = output_size

        self.embedding = VisionEmbedding(output_size, input_size, patch_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(drop_prob)
        self.encoder = TransformerSequential(
            output_size, heads, hidden_size, norm_first=True,
            ff_kwargs=dict(act=nn.GELU()),
            num_blocks=depth
        )

    def forward(self, x):
        x = self.embeddings(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.encoder(x)
        return x


class VisionEmbedding(nn.Module):
    def __init__(self, embed_dim, image_size, patch_size):
        super().__init__()
        self.patch = PatchEmbedding(embed_dim, patch_size)
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
