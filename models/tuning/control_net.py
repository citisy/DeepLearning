from functools import partial

import torch
from torch import nn

from utils import torch_utils
from ..image_generation import ldm
from ..layers import Conv, Linear


class Config(ldm.Config):
    pass


class ModelWrap:
    """only for diffusion models tuning
    https://github.com/lllyasviel/ControlNet?tab=readme-ov-file"""

    def __init__(self, control_model_kwargs):
        self.control_model_kwargs = control_model_kwargs

    def wrap(self, model: nn.Module):
        torch_utils.ModuleManager.freeze_module(model, allow_train=True)
        control_model = ControlBlock(**self.control_model_kwargs)
        control_model = control_model.to(
            device=torch_utils.ModuleInfo.possible_device(model),
            dtype=torch_utils.ModuleInfo.possible_dtype(model)
        )
        model.backbone.control_model = control_model

        self.backbone_forward = model.backbone.forward  # used to restore
        model.backbone.forward = partial(UNetModel.forward, model.backbone)

        self.model = model
        return model

    def dewrap(self):
        self.model.backbone.forward = self.backbone_forward
        del self.model.control_model

    def load_state_dict(self, state_dict, strict=False, **kwargs):
        state_dict = {k.replace('backbone.control_model.', ''): v for k, v in state_dict.items()}
        self.model.backbone.control_model.load_state_dict(state_dict, strict=strict)


class UNetModel(ldm.UNetModel):
    def forward(self, x, timesteps=None, context=None, y=None, control_images=None, only_mid_control=False, **kwargs):
        controls = self.control_model(x, control_images, timesteps, context)

        with torch.no_grad():
            emb = self.time_embed(timesteps)

            if self.num_classes is not None:
                assert y.shape[0] == x.shape[0], f'Expect {y.shape[0]}, but got {x.shape[0]}'
                emb = emb + self.label_emb(y)

            h = x
            hs = []
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)

            h = self.middle_block(h, emb, context)

        if controls is not None:
            h += controls.pop()

        for module in self.output_blocks:
            if only_mid_control or controls is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + controls.pop()], dim=1)
            h = module(h, emb, context)

        return self.out(h)


class ControlBlock(nn.Module):
    def __init__(
            self,
            hint_in_ch=3, in_ch=4, context_dim=768,
            out_ch=4, unit_dim=320, ch_mult=(1, 2, 4, 4),  # for model
            use_checkpoint=True, attend_type=Config.SCALE_ATTEND,  # for resnet and transformers
            groups=32, num_res_blocks=2,  # for resnet
            num_heads=None, head_dim=None, transformer_depth=1, use_linear_in_transformer=False, attend_layers=(0, 1, 2),  # for transformers
            conv_resample=True,  # for up/down sample
    ):
        super().__init__()

        self.in_channels = in_ch
        self.out_channels = out_ch

        time_emb_dim = unit_dim * 4
        self.time_embed = nn.Sequential(
            ldm.SinusoidalPosEmb(unit_dim),
            Linear(unit_dim, time_emb_dim, mode='la', act=nn.SiLU()),
            Linear(time_emb_dim, time_emb_dim, mode='l'),
        )

        zero_layers = [self.make_zero_conv(unit_dim)]

        hint_out_ches = (16, 16, 32, 32, 96, 96, 256)
        hit_strides = (1, 1, 2, 1, 2, 1, 2)

        layers = []
        for hint_out_ch, hit_stride in zip(hint_out_ches, hit_strides):
            layers.append(Conv(hint_in_ch, hint_out_ch, 3, s=hit_stride, mode='ca', act=nn.SiLU()))
            hint_in_ch = hint_out_ch
        layers.append(self.zero_module(Conv(hint_in_ch, unit_dim, 3, mode='c')))
        self.input_hint_block = ldm.TimestepEmbedSequential(*layers)

        if isinstance(transformer_depth, int):
            transformer_depth = len(ch_mult) * [transformer_depth]

        # helper
        make_res = partial(ldm.ResnetBlock, groups=groups, use_checkpoint=use_checkpoint, time_emb_dim=time_emb_dim)
        make_trans = partial(ldm.TransformerBlock, context_dim=context_dim, use_checkpoint=use_checkpoint, use_linear=use_linear_in_transformer, attend_type=attend_type)

        out_ch = unit_dim
        input_layers = [ldm.TimestepEmbedSequential(nn.Conv2d(in_ch, out_ch, 3, padding=1))]
        input_block_chans = [out_ch]

        num_stages = len(ch_mult)
        in_ch = out_ch
        for i, mult in enumerate(ch_mult):
            is_bottom = i == num_stages - 1

            for _ in range(num_res_blocks):
                out_ch = mult * unit_dim
                blocks = [make_res(in_ch, out_ch)]
                if i in attend_layers:
                    _n_heads, _, _head_dim = ldm.get_attention_input(num_heads, out_ch, head_dim)
                    blocks.append(make_trans(out_ch, _n_heads, _head_dim, depth=transformer_depth[i]))

                input_layers.append(ldm.TimestepEmbedSequential(*blocks))
                zero_layers.append(self.make_zero_conv(out_ch))
                input_block_chans.append(out_ch)
                in_ch = out_ch
            if not is_bottom:
                input_layers.append(ldm.TimestepEmbedSequential(
                    ldm.Downsample(out_ch, out_ch, use_conv=conv_resample)
                ))
                zero_layers.append(self.make_zero_conv(out_ch))
                input_block_chans.append(out_ch)

        self.input_blocks = nn.ModuleList(input_layers)
        self.zero_convs = nn.ModuleList(zero_layers)

        _n_heads, _, _head_dim = ldm.get_attention_input(num_heads, out_ch, head_dim)
        self.middle_block = ldm.TimestepEmbedSequential(
            make_res(out_ch, out_ch),
            make_trans(out_ch, _n_heads, _head_dim, depth=transformer_depth[-1]),
            make_res(out_ch, out_ch),
        )

        self.middle_block_out = self.make_zero_conv(out_ch)

    def make_zero_conv(self, ch):
        return ldm.TimestepEmbedSequential(self.zero_module(nn.Conv2d(ch, ch, 1, padding=0)))

    def zero_module(self, module):
        """
        Zero out the parameters of a module and return it.
        """
        for p in module.parameters():
            p.detach().zero_()
        return module

    def initialize_layers(self):
        pass

    def forward(self, x, hint, timesteps, context, **kwargs):
        emb = self.time_embed(timesteps)
        guided_hint = self.input_hint_block(hint, emb, context)

        h = x
        controls = []
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                guided_hint = None
            else:
                h = module(h, emb, context)
            controls.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        controls.append(self.middle_block_out(h, emb, context))

        return controls
