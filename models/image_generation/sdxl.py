import math
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
    FPS = 5
    FPS_ID = 6
    MOTION_BUCKET_ID = 7
    POOL_IMAGE = 8
    COND_AUG = 9
    COND_FRAMES = 10
    COND_FRAMES_WITHOUT_NOISE = 11

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

    def post_process(self, x=None, text=None, neg_text=None, image=None, **kwargs):
        pass


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

    def forward(self, **value_dict):
        batch, batch_uc = self.get_batch(**value_dict)
        c = self.get_cond(batch)
        uc = self.get_cond(batch_uc)
        return c, uc

    def get_cond(self, batch):
        output = {}
        for input_key, embedder in zip(self.input_keys, self.embedders):
            emb_out = embedder(batch[input_key])
            if not isinstance(emb_out, (list, tuple)):
                emb_out = [emb_out]

            for emb in emb_out:
                out_key = self.OUTPUT_DIM2KEYS[emb.dim()]

                if out_key in output:
                    output[out_key] = torch.cat(
                        (output[out_key], emb), self.KEY2CATDIM[out_key]
                    )
                else:
                    output[out_key] = emb

        return output

    def get_batch(self, N, device, **value_dict):
        """
        value_dict = {
        'target_width': 1024,
        'target_height': 1024,
        'crop_coords_top': 0,
        'crop_coords_left': 0,
        'orig_width': 1024,
        'orig_height': 1024,
        'prompt': 'Astronaut in a jungle, cold color palette, muted colors, detailed, 8k',
        'negative_prompt': ''
        }
        """
        batch = {}
        batch_uc = {}
        for key in self.input_keys:
            if key == Config.TXT:
                batch["txt"] = [value_dict["prompt"]] * math.prod(batch)
                batch_uc["txt"] = [value_dict["negative_prompt"]] * math.prod(N)

            elif key == Config.ORIGINAL_SIZE_AS_TUPLE:
                batch[key] = (
                    torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                    .to(device)
                    .repeat(math.prod(N), 1)
                )
            elif key == Config.CROP_COORDS_TOP_LEFT:
                batch[key] = (
                    torch.tensor(
                        [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                    )
                    .to(device)
                    .repeat(math.prod(N), 1)
                )
            elif key == Config.AESTHETIC_SCORE:
                batch[key] = (
                    torch.tensor([value_dict["aesthetic_score"]])
                    .to(device)
                    .repeat(math.prod(N), 1)
                )
                batch_uc[key] = (
                    torch.tensor([value_dict["negative_aesthetic_score"]])
                    .to(device)
                    .repeat(math.prod(N), 1)
                )

            elif key == Config.TARGET_SIZE_AS_TUPLE:
                batch[key] = (
                    torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                    .to(device)
                    .repeat(math.prod(N), 1)
                )
            elif key == Config.FPS:
                batch[key] = (
                    torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
                )
            elif key == Config.FPS_ID:
                batch[key] = (
                    torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
                )
            elif key == Config.MOTION_BUCKET_ID:
                batch[key] = (
                    torch.tensor([value_dict["motion_bucket_id"]])
                    .to(device)
                    .repeat(math.prod(N))
                )
            elif key == Config.POOL_IMAGE:
                batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(
                    device, dtype=torch.half
                )
            elif key == Config.COND_AUG:
                batch[key] = repeat(
                    torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                    "1 -> b",
                    b=math.prod(N),
                )
            elif key == Config.COND_FRAMES:
                batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
            elif key == Config.COND_FRAMES_WITHOUT_NOISE:
                batch[key] = repeat(
                    value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
                )
            else:
                batch[key] = value_dict[key]

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
        b, dims = x.shape[0], x.shape[1]
        x = rearrange(x, "b d -> (b d)")
        emb = self.timestep(x)
        emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return emb
