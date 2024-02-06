import open_clip
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from utils import torch_utils
from . import ldm, ddpm, sdv1, sdv2


class Config(sdv2.Config):
    """only for inference"""

    # for EmbedderWarp input_key
    TXT = 0
    ORIGINAL_SIZE_AS_TUPLE = 1
    CROP_COORDS_TOP_LEFT = 2
    TARGET_SIZE_AS_TUPLE = 3
    AESTHETIC_SCORE = 4

    xl_model = dict(
        scale=9.,
        scale_factor=0.13025,
        objective=ddpm.Config.PRED_Z,
    )

    embedder1 = dict(
        target='models.image_generation.sdv1.CLIPEmbedder',
        input_key=TXT,
        params=dict(
            layer=sdv1.Config.HIDDEN,
            layer_idx=11,
            pretrain_model='/HDD2/lzc/stable-diffusion/openai/clip-vit-large-patch14',
        )
    )

    embedder2 = dict(
        target='models.image_generation.sdv2.OpenCLIPEmbedder',
        input_key=TXT,
        params=dict(
            arch='ViT-bigG-14',
            legacy=False
        )
    )
    embedder3 = dict(
        target='models.image_generation.sdv2.ConcatTimestepEmbedderND',
        input_key=ORIGINAL_SIZE_AS_TUPLE,
        params=dict()
    )

    embedder4 = dict(
        target='models.image_generation.sdxl.ConcatTimestepEmbedderND',
        input_key=CROP_COORDS_TOP_LEFT,
        params=dict()
    )

    embedder5 = dict(
        target='models.image_generation.sdxl.ConcatTimestepEmbedderND',
        input_key=TARGET_SIZE_AS_TUPLE,
        params=dict()
    )

    embedder6 = dict(
        target='models.image_generation.sdxl.ConcatTimestepEmbedderND',
        input_key=AESTHETIC_SCORE,
        params=dict()
    )

    xl_base_cond = [
        embedder1,
        embedder2,
        embedder3,
        embedder4,
        embedder5,
    ]

    xl_refiner_cond = [
        embedder1,
        embedder2,
        embedder3,
        embedder4,
        embedder6,
    ]

    xl_base_backbone = dict(
        num_classes=ldm.Config.SEQUENTIAL,
        adm_in_channels=2816,
        ch_mult=(1, 2, 4),
        # note, for reduce computation, the first layer do not use attention,
        # but use more attention in the middle block
        attend_layers=(1, 2),
        transformer_depth=(1, 2, 10),
    )

    xl_refiner_backbone = dict(
        num_classes=ldm.Config.SEQUENTIAL,
        adm_in_channels=2560,
        unit_dim=384,
        ch_mult=(1, 2, 4, 4),
        attend_layers=(1, 2),
        transformer_depth=4,
    )

    default_model = 'xl_base'

    @classmethod
    def make_full_config(cls):
        config_dict = dict(
            # support sdxl-base-*
            xl_base=dict(
                model_config=cls.xl_model,
                cond_config=cls.xl_base_cond,
                vae_config=cls.v2_unclip_vae,
                backbone_config=cls.xl_base_backbone
            ),

            # support sdxl-refiner-*
            xl_refiner=dict(
                model_config=cls.xl_model,
                cond_config=cls.xl_refiner_cond,
                vae_config=cls.v2_unclip_vae,
                backbone_config=cls.xl_refiner_backbone
            )
        )
        return config_dict


def convert_weights(state_dict):
    state_dict = convert_weights(state_dict)

    convert_dict = {
        'conditioner': 'cond'
    }

    state_dict = torch_utils.convert_state_dict(state_dict, convert_dict)

    return state_dict


class Model(ldm.Model):
    """https://github.com/Stability-AI/generative-models"""

    def make_cond(self, cond_config=[], **kwargs):
        return EmbedderWarp(cond_config)


class EmbedderWarp(nn.Module):
    def __init__(self, cond_configs: list):
        super().__init__()
        from utils import converter

        embedders = []
        input_keys = []
        output_size = 0
        for cond_config in cond_configs:
            ins = converter.InsConvert.str_to_instance(cond_config['target'])
            layer = ins(**cond_config['params'])
            input_key = cond_config['input_key']
            embedders.append(layer)
            input_keys.append(input_key)
            if input_key == Config.TXT:
                output_size += layer.output_size

        self.embedders = nn.ModuleList(embedders)
        self.input_keys = input_keys
        self.output_size = output_size  # 2048

    def forward(self, batch):
        for input_key, embedder in zip(self.input_keys, self.embedders):
            emb_out = embedder(batch[input_key])


class ConcatTimestepEmbedderND(nn.Module):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, output_size=256):
        super().__init__()
        self.timestep = ldm.SinusoidalPosEmb(output_size)
        self.output_size = output_size

    def forward(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return emb
