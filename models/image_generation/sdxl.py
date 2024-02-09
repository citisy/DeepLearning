import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from utils import torch_utils
from . import ldm, ddpm, ddim, sdv1, sdv2
# from .ldm import convert_weights
from typing import Union, Dict, Tuple


class Config(sdv2.Config):
    """only for inference"""

    # for EmbedderWarp input_key
    TXT = 'txt'
    ORIGINAL_SIZE_AS_TUPLE = 'original_size_as_tuple'
    CROP_COORDS_TOP_LEFT = 'crop_coords_top_left'
    TARGET_SIZE_AS_TUPLE = 'target_size_as_tuple'
    AESTHETIC_SCORE = 'aesthetic_score'
    FPS = 'fps'
    FPS_ID = 'fps_id'
    MOTION_BUCKET_ID = 'motion_bucket_id'
    POOL_IMAGE = 'pool_image'
    COND_AUG = 'cond_aug'
    COND_FRAMES = 'cond_frames'
    COND_FRAMES_WITHOUT_NOISE = 'cond_frames_without_noise'

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
        target='models.image_generation.sdxl.ConcatTimestepEmbedderND',
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
        embedder2,
        embedder3,
        embedder4,
        embedder6,
    ]

    xl_base_backbone = dict(
        num_classes=ldm.Config.SEQUENTIAL,
        adm_in_channels=2816,   # 1028 * 2 + 256 * 3
        ch_mult=(1, 2, 4),
        # note, for reduce computation, the first layer do not use attention,
        # but use more attention in the middle block
        attend_layers=(1, 2),
        transformer_depth=(1, 2, 10),
        head_dim=64,
        use_linear_in_transformer=True
    )

    xl_refiner_backbone = dict(
        num_classes=ldm.Config.SEQUENTIAL,
        adm_in_channels=2560,   # 1028 * 2 + 256 * 2
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
                vae_config=cls.vae,
                backbone_config=cls.xl_base_backbone
            ),

            # support sdxl-refiner-*
            xl_refiner=dict(
                model_config=cls.xl_model,
                cond_config=cls.xl_refiner_cond,
                vae_config=cls.vae,
                backbone_config=cls.xl_refiner_backbone
            )
        )
        return config_dict


def convert_weights(state_dict):
    state_dict = ldm.convert_weights(state_dict)

    convert_dict = {
        'conditioner': 'cond'
    }

    state_dict = torch_utils.convert_state_dict(state_dict, convert_dict)

    return state_dict


class Model(ldm.Model):
    """https://github.com/Stability-AI/generative-models"""

    def make_cond(self, cond_config=[], **kwargs):
        return EmbedderWarp(cond_config)

    def make_txt_cond(self, text, neg_text=None) -> dict:
        if not neg_text:
            neg_text = [''] * len(text)

        value_dicts = [
            {
                'prompt': prompt,
                'negative_prompt': negative_prompt,
                Config.ORIGINAL_SIZE_AS_TUPLE: self.image_size,
                Config.CROP_COORDS_TOP_LEFT: (0, 0),
                Config.TARGET_SIZE_AS_TUPLE: self.image_size,
            } for prompt, negative_prompt in zip(text, neg_text)
        ]

        c_values, uc_values = self.cond(value_dicts)

        if self.scale == 1.0:
            uc_values = None

        return dict(
            c_values=c_values,
            uc_values=uc_values
        )

    def diffuse(self, x, time, c_values=None, uc_values=None, **kwargs):
        if uc_values is not None:
            x = torch.cat([x] * 2)
            time = torch.cat([time] * 2)
            cond = torch.cat([uc_values['crossattn'], c_values['crossattn']])
            y = torch.cat([uc_values['vector'], c_values['vector']])
        else:
            cond = c_values['crossattn']
            y = c_values['vector']

        z = self.backbone(x, timesteps=time, context=cond, y=y)
        if uc_values is None:
            e_t = z
        else:
            e_t_uncond, e_t = z.chunk(2)
            e_t = e_t_uncond + self.scale * (e_t - e_t_uncond)

        return e_t


class EmbedderWarp(nn.Module):
    OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
    KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}

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
        self.dummy_params = nn.Parameter(torch.empty(0))

    def forward(self, value_dicts):
        batch, batch_uc = self.get_batch(value_dicts)
        c_values = self.get_cond(batch)
        uc_values = self.get_cond(batch_uc, [Config.TXT])
        return c_values, uc_values

    def get_cond(self, batch, force_zero_embeddings=()):
        output = {}
        for input_key, embedder in zip(self.input_keys, self.embedders):
            emb_out = embedder(batch[input_key])
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]

            for emb in emb_out:
                if input_key in force_zero_embeddings:
                    emb = torch.zeros_like(emb)

                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]
                output.setdefault(out_key, []).append(emb)

        for out_key, v in output.items():
            output[out_key] = torch.cat(v, self.KEY2CATDIM[out_key])

        return output

    def get_batch(self, value_dicts):
        device = self.dummy_params.device
        batch = {}
        batch_uc = {}

        for value_dict in value_dicts:
            for key in self.input_keys:
                if key == Config.TXT:
                    pass

                elif key == Config.POOL_IMAGE:
                    batch.setdefault(key, []).append(value_dict[key].to(dtype=torch.half))

                else:
                    batch.setdefault(key, []).append(torch.tensor(value_dict[key]))

        for k, v in batch_uc.items():
            batch_uc[k] = torch.stack(v).to(device)

        for k, v in batch.items():
            batch[k] = torch.stack(v).to(device)

            if k not in batch_uc:
                batch_uc[k] = torch.clone(batch[k])

        batch[Config.TXT] = [value_dict["prompt"] for value_dict in value_dicts]
        batch_uc[Config.TXT] = [value_dict["negative_prompt"] for value_dict in value_dicts]

        return batch, batch_uc


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
        b, dims = x.shape
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.output_size)
        return emb
