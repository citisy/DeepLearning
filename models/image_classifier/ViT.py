import torch
from torch import nn
from . import BaseImgClsModel
from ..layers import OutModule
from einops.layers.torch import Rearrange

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
        neck = Neck()
        head = out_module or OutModule(out_features, in_features=backbone.output_size)

        super().__init__(
            in_module=nn.Identity(),  # placeholder
            backbone=backbone,
            neck=neck,
            head=head,
            **kwargs
        )


class Backbone(nn.Module):
    def __init__(self, in_ch=3, input_size=256, output_size=1024,
                 patch_size=32, hidden_size=2048, depth=6, heads=8, dropout=0.1, emb_dropout=0.1):
        super().__init__()

        self.in_ch = in_ch
        self.input_size = input_size
        self.output_size = output_size

        num_patches = (input_size // patch_size) ** 2
        p_size = in_ch * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(p_size),
            nn.Linear(p_size, output_size),
            nn.LayerNorm(output_size),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, output_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, output_size))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            output_size, heads, dim_feedforward=hidden_size, dropout=dropout,
            activation=nn.GELU(), batch_first=True, norm_first=True,
        ), depth)

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = self.cls_token.repeat(b, 1, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return x


class Neck(nn.Module):
    def forward(self, x):
        return x[:, 0]
