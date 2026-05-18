from utils import torch_utils
from . import Qwen2_VL
from .. import bundles


class Config(bundles.Config):
    _2_5_vit_config = dict(
        norm_type='RMSNorm2D',
        ff_type='GateFeedForward'
    )

    _3b_vit_config = dict(
        output_size=2048,
        ff_hidden_size=3420,
        **_2_5_vit_config
    )

    _3b_vlm_config = dict(
        hidden_size=2048,
        ff_hidden_size=11008,
        num_heads=16,
        num_blocks=36,
        num_kv_heads=2,
        vocab_size=151936
    )

    _7b_vit_config = dict(
        output_size=3584,
        ff_hidden_size=3420,
        **_2_5_vit_config
    )

    _72b_vit_config = dict(
        output_size=8192,
        ff_hidden_size=3456,
        **_2_5_vit_config
    )

    default_model = '3b'

    @classmethod
    def make_full_config(cls):
        return {
            '3b': dict(
                vit_config=cls._3b_vit_config,
                vlm_config=cls._3b_vlm_config
            ),

            '7b': dict(
                vit_config=cls._7b_vit_config,
                vlm_config=Qwen2_VL.Config._7b_vlm_config
            ),

            '72b': dict(
                vit_config=cls._72b_vit_config,
                vlm_config=Qwen2_VL.Config._72b_vlm_config
            )
        }


class WeightLoader(bundles.WeightLoader):
    pass


class WeightConverter:
    vit_convert_dict = {
        'visual': 'vit',
        'visual.patch_embed.proj': 'vit.patch_embed.fn',
        'visual.blocks.{0}.norm1': 'vit.blocks.{0}.attn_res.norm',
        'visual.blocks.{0}.norm2': 'vit.blocks.{0}.ff_res.norm',
        'visual.blocks.{0}.attn.qkv': 'vit.blocks.{0}.attn_res.fn.to_qkv',
        'visual.blocks.{0}.attn.proj': 'vit.blocks.{0}.attn_res.fn.to_out.linear',
        'visual.blocks.{0}.mlp.down_proj': 'vit.blocks.{0}.ff_res.fn.f2.linear',
        'visual.blocks.{0}.mlp.gate_proj': 'vit.blocks.{0}.ff_res.fn.f1.linear',
        'visual.blocks.{0}.mlp.up_proj': 'vit.blocks.{0}.ff_res.fn.f3.linear',
    }

    convert_dict = {
        **vit_convert_dict,
        **Qwen2_VL.WeightConverter.vlm_convert_dict,
        'lm_head': 'head'
    }

    @classmethod
    def from_official(cls, state_dict):
        state_dict = torch_utils.Converter.convert_keys(state_dict, cls.convert_dict)
        return state_dict


class Model(Qwen2_VL.Model):
    """https://github.com/QwenLM/Qwen3-VL"""

    def __init__(self, vit_config=Config._3b_vit_config, vlm_config=Config._3b_vlm_config, model_config={}):  # noqa
        super().__init__(vit_config=vit_config, vlm_config=vlm_config, model_config=model_config)
