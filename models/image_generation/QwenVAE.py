from collections import OrderedDict

from utils import torch_utils
from . import VAE
import torch
from torch import nn
import torch.nn.functional as F
from .. import normalizations, attentions, bundles
from einops import rearrange


class Config(bundles.Config):
    vae = dict(
        backbone_config=dict(
            z_ch=16,
            unit_ch=96,
            ch_mult=(1, 2, 4, 4),
            attn_layers=[]
        )
    )


class WeightConverter:
    convert_dict = {
        'decoder.up_blocks.{0}.resnets.{1}.norm1': 'decoder.up.{0}.layers.{1}.fn.0',
        'decoder.up_blocks.{0}.resnets.{1}.conv1': 'decoder.up.{0}.layers.{1}.fn.2',
        'decoder.up_blocks.{0}.resnets.{1}.norm2': 'decoder.up.{0}.layers.{1}.fn.3',
        'decoder.up_blocks.{0}.resnets.{1}.conv2': 'decoder.up.{0}.layers.{1}.fn.6',
        'decoder.up_blocks.{0}.resnets.{1}.conv_shortcut': 'decoder.up.{0}.layers.{1}.proj',
        'decoder.up_blocks.{0}.upsamplers.0.resample.1': 'decoder.up.{0}.head.resample.1',
        'decoder.up_blocks.{0}.upsamplers.0.time_conv': 'decoder.up.{0}.head.time_conv',

        'encoder.down_blocks.{1}.norm1': 'encoder.down.{1}.fn.0',
        'encoder.down_blocks.{1}.conv1': 'encoder.down.{1}.fn.2',
        'encoder.down_blocks.{1}.norm2': 'encoder.down.{1}.fn.3',
        'encoder.down_blocks.{1}.conv2': 'encoder.down.{1}.fn.6',
        'encoder.down_blocks.{1}.resample.1': 'encoder.down.{1}.resample.1',
        'encoder.down_blocks.{1}.conv_shortcut': 'encoder.down.{1}.proj',
        'encoder.down_blocks.{1}.time_conv': 'encoder.down.{1}.time_conv',

        '{0}.mid_block.resnets.{1}.norm1': '{0}.neck.{0 if [1]==0 else 2}.fn.0',
        '{0}.mid_block.resnets.{1}.conv1': '{0}.neck.{0 if [1]==0 else 2}.fn.2',
        '{0}.mid_block.resnets.{1}.norm2': '{0}.neck.{0 if [1]==0 else 2}.fn.3',
        '{0}.mid_block.resnets.{1}.conv2': '{0}.neck.{0 if [1]==0 else 2}.fn.6',
        '{0}.mid_block.attentions.{1}.norm': '{0}.neck.{[1]+1}.0',
        '{0}.mid_block.attentions.{1}.to_qkv': '{0}.neck.{[1]+1}.1.to_qkv',
        '{0}.mid_block.attentions.{1}.proj': '{0}.neck.{[1]+1}.1.to_out',

        '{0}.norm_out': '{0}.head.0',
        '{0}.conv_out': '{0}.head.2',
    }

    @classmethod
    def from_diffusers(cls, state_dict):
        """weights trained from `diffusers`"""
        state_dict = torch_utils.Converter.convert_keys(state_dict, cls.convert_dict)
        state_dict = OrderedDict({k.replace('gamma', 'weight'): v for k, v in state_dict.items()})
        return state_dict


class Model(VAE.Model):
    scale_factor = 1 / torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
    ])[None, :, None, None, None]
    shift_factor = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
    ])[None, :, None, None, None]

    def set_encoder(self, **backbone_config):
        self.encoder = Encoder3d(self.img_ch, **backbone_config)
        self.quant_conv = CausalConv3d(self.encoder.z_ch * 2, self.encoder.z_ch * 2, 1) if self.use_quant_conv else nn.Identity()

    def set_decoder(self, **backbone_config):
        z_ch = self.encoder.z_ch
        self.post_quant_conv = CausalConv3d(z_ch, z_ch, 1) if self.use_post_quant_conv else nn.Identity()
        self.decoder = Decoder3d(z_ch, self.img_ch, **backbone_config)

    @staticmethod
    def count_conv3d(model):
        count = 0
        for m in model.modules():
            if isinstance(m, CausalConv3d):
                count += 1
        return count

    def encode(self, x, sample_posterior=True):
        feat_cache = [None] * self.count_conv3d(self.encoder)

        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4

        out = None
        for i in range(iter_):
            feat_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=feat_cache,
                    feat_idx=feat_idx
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=feat_cache,
                    feat_idx=feat_idx
                )
                out = torch.cat([out, out_], 2)

        out = self.quant_conv(out)
        z, mean, log_var = self.re_parametrize(out, sample_posterior=sample_posterior)

        scale_factor = self.scale_factor
        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.to(z)
        shift_factor = self.shift_factor
        if isinstance(shift_factor, torch.Tensor):
            shift_factor = shift_factor.to(z)

        z = scale_factor * (z - shift_factor)
        return z, mean, log_var

    def decode(self, z):
        feat_cache = [None] * self.count_conv3d(self.decoder)

        scale_factor = self.scale_factor
        if isinstance(scale_factor, torch.Tensor):
            scale_factor = scale_factor.to(z)
        shift_factor = self.shift_factor
        if isinstance(shift_factor, torch.Tensor):
            shift_factor = shift_factor.to(z)

        z = z / scale_factor + shift_factor

        iter_ = z.shape[2]
        x = self.post_quant_conv(z)
        out = None
        for i in range(iter_):
            feat_idx = [0]
            out_ = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=feat_cache,
                feat_idx=feat_idx
            )
            if i == 0:
                out = out_
            else:
                out = torch.cat([out, out_], 2)
        return out


class Encoder3d(nn.Module):
    def __init__(
            self, in_ch, unit_ch=128, z_ch=64,
            ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2, attn_layers=(-1, -2),
            drop_prob=0.0, double_z=True,
            **ignore_kwargs
    ):
        super().__init__()
        num_layers = len(ch_mult)

        self.conv_in = CausalConv3d(in_ch, unit_ch, 3, stride=1, padding=1)

        in_ch = unit_ch
        down = []
        for i in range(num_layers):
            is_top = i == num_layers - 1
            is_2d = i == 0  # the first layers
            out_ch = unit_ch * ch_mult[i]

            for _ in range(num_res_blocks):
                down.append(ResBlock(in_ch, out_ch, drop_prob=drop_prob))
                if i in attn_layers:
                    down.append(AttentionBlock(out_ch))

                in_ch = out_ch

            if not is_top:
                down.append(DownSample(in_ch, is_2d))
        self.down = nn.ModuleList(down)

        self.neck = nn.ModuleList([
            ResBlock(in_ch, in_ch, drop_prob=drop_prob),
            AttentionBlock(in_ch),
            ResBlock(in_ch, in_ch, drop_prob=drop_prob)
        ])

        out_ch = 2 * z_ch if double_z else z_ch
        self.head = nn.ModuleList([
            normalizations.RMSNorm4D(in_ch),
            nn.SiLU(),
            CausalConv3d(in_ch, out_ch, 3, padding=1),
        ])

        self.out_channels = out_ch
        self.down_scale = 2 ** (num_layers - 1)
        self.z_ch = z_ch

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -2:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.forward_down(x, feat_cache, feat_idx)
        x = self.forward_up(x, feat_cache, feat_idx)
        x = self.forward_head(x, feat_cache, feat_idx)

        return x

    def forward_down(self, x, feat_cache, feat_idx):
        for layer in self.down:
            if not isinstance(layer, AttentionBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            elif isinstance(layer, AttentionBlock):
                b, c, t, h, w = x.shape
                x = rearrange(x, 'b c t h w -> (b t) c h w')
                x = layer(x)
                x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
            else:
                x = layer(x)
        return x

    def forward_neck(self, x, feat_cache, feat_idx):
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -2:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x

    def forward_head(self, x, feat_cache, feat_idx):
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -2:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):
    def __init__(
            self, in_ch, out_ch, unit_ch=128,
            ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2, attn_layers=(-1, -2),
            drop_prob=0.0,
            **ignore_kwargs
    ):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        num_layers = len(ch_mult)
        attn_layers = [i % num_layers for i in attn_layers]

        # z to block_in
        in_ch = unit_ch * ch_mult[num_layers - 1]
        self.conv_in = CausalConv3d(self.in_channels, in_ch, 3, stride=1, padding=1)

        # middle
        self.neck = nn.ModuleList([
            ResBlock(in_ch, in_ch, drop_prob=drop_prob),
            AttentionBlock(in_ch),
            ResBlock(in_ch, in_ch, drop_prob=drop_prob)
        ])

        # upsample
        up = []
        for i in reversed(range(num_layers)):
            is_bottom = i == 0
            is_top = i == num_layers - 1
            is_2d = i == 1  # the first 2 layers
            if not is_top:
                in_ch //= 2
            out_ch = unit_ch * ch_mult[i]
            up.append(UpBlock(
                in_ch, out_ch, num_res_blocks, is_bottom, i, is_2d,
                attn_layers=attn_layers, drop_prob=drop_prob,
            ))
            in_ch = out_ch
        # note, implement different from ldm
        self.up = nn.ModuleList(up)

        self.head = nn.ModuleList([
            normalizations.RMSNorm4D(in_ch),
            nn.SiLU(),
            CausalConv3d(in_ch, self.out_channels, 3, padding=1),
        ])

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -2:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.forward_neck(x, feat_cache, feat_idx)
        x = self.forward_up(x, feat_cache, feat_idx)
        x = self.forward_head(x, feat_cache, feat_idx)
        return x

    def forward_neck(self, x, feat_cache, feat_idx):
        for layer in self.neck:
            if isinstance(layer, ResBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            elif isinstance(layer, AttentionBlock):
                b, c, t, h, w = x.shape
                x = rearrange(x, 'b c t h w -> (b t) c h w')
                x = layer(x)
                x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
            else:
                x = layer(x)

        return x

    def forward_up(self, x, feat_cache, feat_idx):
        for layer in self.up:
            x = layer(x, feat_cache, feat_idx)
        return x

    def forward_head(self, x, feat_cache, feat_idx):
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -2:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch=None, conv_shortcut=False, drop_prob=0.):
        super().__init__()
        out_ch = in_ch if out_ch is None else out_ch
        self.in_channels = in_ch
        self.out_channels = out_ch

        if in_ch != out_ch:
            if conv_shortcut:
                shortcut = CausalConv3d(in_ch, out_ch, 3, stride=1, padding=1)
            else:
                shortcut = CausalConv3d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            shortcut = nn.Identity()

        self.proj = shortcut
        self.fn = ResFn(in_ch, out_ch, drop_prob=drop_prob)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h1 = self.proj(x)
        h2 = self.fn(x, feat_cache, feat_idx)
        return h1 + h2


class ResFn(nn.ModuleList):
    def __init__(self, in_ch, out_ch, drop_prob=0.):
        super().__init__([
            normalizations.RMSNorm4D(in_ch),
            nn.SiLU(),
            CausalConv3d(in_ch, out_ch, 3, padding=1),

            normalizations.RMSNorm4D(out_ch),
            nn.SiLU(),
            nn.Dropout(drop_prob),
            CausalConv3d(out_ch, out_ch, 3, padding=1)
        ])

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        for layer in self:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -2:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class AttentionBlock(nn.Sequential):
    def __init__(self, in_ch):
        super().__init__(
            normalizations.RMSNorm3D(in_ch),
            attentions.CrossAttention3D(
                n_heads=1, head_dim=in_ch, separate=False,
                # attend=attentions.SplitScaleAttend()
            )
        )

    def forward(self, x):
        h = super().forward(x)
        return x + h


class DownSample(nn.Module):
    def __init__(self, in_ch, is_2d=True):
        super().__init__()
        self.is_2d = is_2d

        self.resample = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(in_ch, in_ch, 3, stride=(2, 2))
        )

        if not is_2d:
            self.time_conv = CausalConv3d(in_ch, in_ch, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if not self.is_2d:
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class UpSample32(nn.Upsample):
    def forward(self, x):
        """forced to use fp32"""
        return super().forward(x.float()).type_as(x)


class UpSample(nn.Module):
    def __init__(self, in_ch, is_2d=True):
        super().__init__()
        self.is_2d = is_2d

        self.resample = nn.Sequential(
            UpSample32(scale_factor=(2., 2.), mode='nearest-exact'),
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1)
        )
        if not is_2d:
            self.time_conv = CausalConv3d(in_ch, in_ch * 2, (3, 1, 1), padding=(1, 0, 0))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if not self.is_2d:
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -2:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x


class UpBlock(nn.Module):
    def __init__(
            self, in_ch, out_ch, num_res_blocks, is_bottom, cur_idx, is_2d,
            attn_layers=(-1, -2), drop_prob=0.0,
    ):
        super().__init__()

        layers = []
        for j in range(num_res_blocks + 1):
            layers.append(ResBlock(in_ch, out_ch, drop_prob=drop_prob))
            if cur_idx in attn_layers:
                layers.append(AttentionBlock(out_ch))
            in_ch = out_ch

        self.layers = nn.ModuleList(layers)
        self.head = nn.Identity() if is_bottom else UpSample(in_ch, is_2d)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        for layer in self.layers:
            if not isinstance(layer, AttentionBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            elif isinstance(layer, AttentionBlock):
                b, c, t, h, w = x.shape
                x = rearrange(x, 'b c t h w -> (b t) c h w')
                x = layer(x)
                x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
            else:
                x = layer(x)
        x = self.head(x)
        return x
