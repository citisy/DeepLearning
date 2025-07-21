from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from utils import torch_utils
from .. import attentions, activations, normalizations
from ..layers import Linear, Conv, Residual


class Config:
    VANILLA = 0
    LINEAR = 1
    VANILLA_XFORMERS = 2

    # config from ldm
    # 8x8x64 gives the size of sampler feature map where image size is 256
    backbone_8x8x64 = dict(
        z_ch=64,
        ch_mult=(1, 1, 2, 2, 4, 4),  # 256 / 2^(6-1) = 8
        attn_layers=[-1, -2],
        tanh_out=True
    )

    backbone_16x16x16 = dict(
        z_ch=16,
        ch_mult=(1, 1, 2, 2, 4),
        attn_layers=[-1]
    )

    backbone_32x32x4 = dict(
        z_ch=4,
        ch_mult=(1, 2, 4, 4),
        attn_layers=[]
    )

    # required pytorch>2.0
    backbone_32x32x4_with_xformers = dict(
        attn_type=VANILLA_XFORMERS,
        **backbone_32x32x4
    )

    backbone_64x64x3 = dict(
        z_ch=3,
        ch_mult=(1, 2, 4),
        attn_layers=[]
    )

    loss = dict(
        use_gan=False,
        use_lpips=False
    )

    @classmethod
    def get(cls, name='backbone_8x8x64'):
        return dict(
            backbone_config=getattr(cls, name),
            loss_config=cls.loss,
        )


class WeightConverter:
    convert_dict = {
        '{0}.block.{1}.norm{2}.': '{0}.blocks.{1}.fn.conv{2}.norm.',
        '{0}.block.{1}.conv{2}.': '{0}.blocks.{1}.fn.conv{2}.conv.',
        '{0}.block.{1}.nin_shortcut': '{0}.blocks.{1}.proj',
        '{0}sample.conv': '{0}sample.fn.1',
        '{0}.mid.block_{1}.norm{2}.': '{0}.neck.block_{1}.fn.conv{2}.norm.',
        '{0}.mid.block_{1}.conv{2}.': '{0}.neck.block_{1}.fn.conv{2}.conv.',
        '{0}.mid.attn_1.norm': '{0}.neck.attn.0',
        '{0}.mid.attn_1.q': '{0}.neck.attn.1.to_qkv.0',
        '{0}.mid.attn_1.k': '{0}.neck.attn.1.to_qkv.1',
        '{0}.mid.attn_1.v': '{0}.neck.attn.1.to_qkv.2',
        '{0}.mid.attn_1.proj_out': '{0}.neck.attn.1.to_out',
        '{0}.norm_out': '{0}.head.norm',
        '{0}.conv_out': '{0}.head.conv',
    }

    @classmethod
    def from_ldm_official(cls, state_dict):
        state_dict = torch_utils.Converter.convert_keys(state_dict, cls.convert_dict)
        return state_dict


class Model(nn.Module):
    """vae from ldm"""

    use_quant_conv = True
    use_post_quant_conv = True

    scale_factor = 1.
    shift_factor = 0.

    def __init__(self, img_ch=3, backbone_config=Config.backbone_8x8x64, loss_config=Config.loss, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        z_ch = backbone_config["z_ch"]

        self.encoder = Encoder(img_ch, **backbone_config)
        self.quant_conv = nn.Conv2d(self.encoder.out_channels, 2 * z_ch, 1) if self.use_quant_conv else nn.Identity()
        self.re_parametrize = ReParametrize()
        self.post_quant_conv = nn.Conv2d(z_ch, z_ch, 1) if self.use_post_quant_conv else nn.Identity()
        self.decoder = Decoder(z_ch, img_ch, **backbone_config)
        self.loss = Loss(self.re_parametrize, **loss_config)
        self.z_ch = z_ch

    def set_inference_only(self):
        del self.loss

    def forward(self, x, sample_posterior=True, **loss_kwargs):
        z, mean, log_var = self.encode(x, sample_posterior)
        z = self.decode(z)

        if self.training:
            return self.loss(x, z, mean, log_var, last_layer=self.decoder.head.conv.weight, **loss_kwargs)
        else:
            return z

    def encode(self, x, sample_posterior=True):
        h = self.encoder(x)
        h = self.quant_conv(h)
        z, mean, log_var = self.re_parametrize(h, sample_posterior=sample_posterior)
        z = self.scale_factor * (z - self.shift_factor)
        return z, mean, log_var

    def decode(self, z):
        z = self.post_quant_conv(z)
        z = z / self.scale_factor + self.shift_factor
        z = self.decoder(z)
        return z


def make_attn(in_channels, attn_type=Config.VANILLA, groups=32):
    attn_dict = {
        Config.VANILLA: lambda in_ch: nn.Sequential(
            make_norm(groups, in_ch),
            attentions.CrossAttention3D(n_heads=1, head_dim=in_ch, attend=attentions.SplitScaleAttend())
        ),
        Config.VANILLA_XFORMERS: lambda in_ch: nn.Sequential(
            make_norm(groups, in_ch),
            attentions.CrossAttention3D(n_heads=1, head_dim=in_ch, attend=attentions.ScaleAttendWithXformers())
        ),  # use xformers, equal to VANILLA
        Config.LINEAR: lambda in_ch: attentions.LinearAttention3D(n_heads=1, head_dim=in_ch, separate=True),
    }
    return attn_dict.get(attn_type, nn.Identity)(in_channels)


make_norm = partial(normalizations.GroupNorm32, eps=1e-6, affine=True)


class ResBlock(Residual):
    def __init__(self, in_ch, out_ch=None, conv_shortcut=False, time_emb_ch=512, drop_prob=0.):
        out_ch = in_ch if out_ch is None else out_ch
        self.in_channels = in_ch
        self.out_channels = out_ch

        if self.in_channels != self.out_channels:
            if conv_shortcut:
                shortcut = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            else:
                shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            shortcut = nn.Identity()

        super().__init__(
            fn=ResFn(in_ch, out_ch, time_emb_ch=time_emb_ch, drop_prob=drop_prob),
            proj=shortcut,
            is_norm=False
        )


class ResFn(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_ch=512, drop_prob=0., groups=32):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch, 3, mode='nac',
                          norm=make_norm(groups, in_ch),
                          act=activations.Swish())
        if time_emb_ch > 0:
            self.time_emb_proj = Linear(time_emb_ch, out_ch, mode='la', act=activations.Swish())

        self.conv2 = Conv(out_ch, out_ch, 3, mode='nadc',
                          norm=make_norm(groups, out_ch),
                          act=activations.Swish(), drop_prob=drop_prob)

    def forward(self, x, time_emb=None):
        h = x
        h = self.conv1(h)

        if time_emb is not None:
            h = h + self.time_emb_proj(time_emb)[:, :, None, None]

        h = self.conv2(h)
        return h


class NeckBlock(nn.Module):
    def __init__(self, in_ch, time_emb_ch, attn_type, drop_prob):
        super().__init__()
        self.block_1 = ResBlock(in_ch=in_ch, out_ch=in_ch, time_emb_ch=time_emb_ch, drop_prob=drop_prob)
        self.attn = make_attn(in_ch, attn_type=attn_type)
        self.block_2 = ResBlock(in_ch=in_ch, out_ch=in_ch, time_emb_ch=time_emb_ch, drop_prob=drop_prob)

    def forward(self, h, time_emb):
        h = self.block_1(h, time_emb=time_emb)
        h = self.attn(h) + h
        h = self.block_2(h, time_emb=time_emb)
        return h


class Encoder(nn.Module):
    def __init__(self, in_ch, unit_ch=128, z_ch=64,
                 ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2, attn_layers=(-1, -2),
                 drop_prob=0.0, resample_with_conv=True, time_emb_ch=0, groups=32,
                 double_z=True, attn_type=Config.VANILLA,
                 **ignore_kwargs):
        super().__init__()
        num_layers = len(ch_mult)
        attn_layers = [i % num_layers for i in attn_layers]

        # down_sampling
        self.conv_in = nn.Conv2d(in_ch, unit_ch, 3, stride=1, padding=1)

        in_ch = unit_ch
        self.down = nn.ModuleList()
        for i in range(num_layers):
            is_top = i == num_layers - 1
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            out_ch = unit_ch * ch_mult[i]
            for j in range(num_res_blocks):
                blocks.append(ResBlock(in_ch=in_ch, out_ch=out_ch, time_emb_ch=time_emb_ch, drop_prob=drop_prob))
                attns.append(make_attn(out_ch, attn_type=attn_type) if i in attn_layers else nn.Identity())
                in_ch = out_ch

            down = nn.Module()
            down.blocks = blocks
            down.attns = attns
            down.downsample = nn.Identity() if is_top else DownSample(in_ch, resample_with_conv)
            self.down.append(down)

        self.neck = NeckBlock(in_ch, time_emb_ch, attn_type, drop_prob)

        out_ch = 2 * z_ch if double_z else z_ch
        self.head = Conv(in_ch, out_ch, 3, mode='nac', norm=make_norm(groups, in_ch), act=activations.Swish())
        self.out_channels = out_ch
        self.down_scale = 2 ** (num_layers - 1)

    def forward(self, x, time_emb=None):
        h = self.conv_in(x)
        for m in self.down:
            for block, attn in zip(m.blocks, m.attns):
                h = block(h, time_emb=time_emb)
                h = attn(h)
            h = m.downsample(h)

        h = self.neck(h, time_emb)
        h = self.head(h)
        return h


class DownSample(nn.Module):
    def __init__(self, in_ch, use_conv=True):
        super().__init__()
        if use_conv:
            self.fn = nn.Sequential(
                nn.ConstantPad2d((0, 1, 0, 1), value=0),  # no asymmetric padding in torch conv, must do it ourselves
                nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=0)
            )
        else:
            self.fn = nn.AvgPool2d(2, 2)

    def forward(self, x):
        x = self.fn(x)
        return x


class ReParametrize(nn.Module):
    """fit mean and log_var for decoder"""

    def __init__(self, deterministic=False):
        super().__init__()
        self.deterministic = deterministic

    def forward(self, x, sample_posterior=True):
        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30.0, 20.0)

        if self.deterministic:
            std = torch.zeros_like(mean, device=x.device)
        else:
            std = torch.exp(0.5 * log_var)

        if sample_posterior:
            z = mean + std * torch.randn(mean.shape, device=x.device)
        else:
            z = mean

        return z, mean, log_var

    def loss(self, mean, log_var, other=None):
        """0.5 * sum(mu^2 + sigma^2 - 1 - log(sigma^2))"""
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            var = torch.exp(log_var)
            if other is None:
                return 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - log_var, dim=[1, 2, 3]).mean()
            else:
                return 0.5 * torch.sum(torch.pow(mean - other.mean, 2) / other.var + var / other.var - 1.0 - log_var + other.log_var, dim=[1, 2, 3]).mean()


class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch, unit_ch=128,
                 ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2, attn_layers=(-1, -2),
                 drop_prob=0.0, resample_with_conv=True,
                 give_pre_end=False, tanh_out=False, groups=32,
                 attn_type=Config.VANILLA, **ignore_kwargs):
        super().__init__()
        time_emb_ch = 0
        self.in_channels = in_ch
        self.out_channels = out_ch
        num_layers = len(ch_mult)
        attn_layers = [i % num_layers for i in attn_layers]

        # z to block_in
        in_ch = unit_ch * ch_mult[num_layers - 1]
        self.conv_in = nn.Conv2d(self.in_channels, in_ch, 3, stride=1, padding=1)

        # middle
        self.neck = NeckBlock(in_ch, time_emb_ch, attn_type, drop_prob)

        # upsampling
        self.up = []
        for i in reversed(range(num_layers)):
            is_bottom = i == 0
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            out_ch = unit_ch * ch_mult[i]
            for j in range(num_res_blocks + 1):
                blocks.append(ResBlock(in_ch=in_ch, out_ch=out_ch, time_emb_ch=time_emb_ch, drop_prob=drop_prob))
                attns.append(make_attn(out_ch, attn_type=attn_type) if i in attn_layers else nn.Identity())
                in_ch = out_ch
            up = nn.Module()
            up.blocks = blocks
            up.attns = attns
            up.upsample = nn.Identity() if is_bottom else Upsample(in_ch, resample_with_conv)
            self.up.append(up)  # prepend to get consistent order

        # note, apply for official ldm code, use reversed layers
        self.up = nn.ModuleList(self.up[::-1])

        # end
        if give_pre_end:
            self.head = nn.Identity()
            self.out_act = nn.Identity()
        else:
            self.head = Conv(in_ch, self.out_channels, 3, mode='nac', norm=make_norm(groups, in_ch), act=activations.Swish())
            self.out_act = nn.Tanh() if tanh_out else nn.Identity()  # todo: something wrong???

    def forward(self, z, time_emb=None):
        h = self.conv_in(z)
        h = self.neck(h, time_emb)

        # upsampling
        for m in reversed(self.up):
            for block, attn in zip(m.blocks, m.attns):
                h = block(h, time_emb=time_emb)
                h = attn(h)
            h = m.upsample(h)

        h = self.head(h)
        h = self.out_act(h)
        return h


class Upsample(nn.Module):
    def __init__(self, in_ch, use_conv=True):
        super().__init__()
        up = nn.Upsample(scale_factor=2.)
        self.fn = nn.Sequential(
            up,
            nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        ) if use_conv else up

    def forward(self, x):
        return self.fn(x)


class Loss(nn.Module):
    def __init__(self, re_parametrize, disc_start=50001, logvar_init=0.0,
                 use_lpips=False, use_gan=False,
                 disc_weight=1.0, kl_weight=1.0, perceptual_weight=1.0, nll_weight=1.,
                 disc_num_layers=3, disc_in_ch=3, disc_factor=1.0,
                 use_actnorm=False, disc_loss="hinge"):

        super().__init__()

        self.re_parametrize = re_parametrize
        self.use_lpips = use_lpips
        self.use_gan = use_gan

        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_iter_start = disc_start
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.nll_weight = nll_weight

        if use_lpips:
            self.perceptual_loss = LPIPS().eval()

        if use_gan:
            from ..losses import HingeGanLoss
            from ..image_translation.pix2pix import NetD as NLayerDiscriminator

            assert disc_loss in ["hinge", "vanilla"]
            self.discriminator = NLayerDiscriminator(disc_in_ch, n_layers=disc_num_layers, norm_layer=ActNorm if use_actnorm else nn.BatchNorm2d)
            self.disc_loss = HingeGanLoss if disc_loss == "hinge" else self.vanilla_d_loss

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        torch_utils.ModuleManager.initialize_layers(self)

    @staticmethod
    def vanilla_d_loss(logits_real, logits_fake):
        d_loss = torch.mean(F.softplus(-logits_real)) + torch.mean(F.softplus(logits_fake))
        return d_loss

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, true, pred, mean, log_var, optimizer_idx=0, global_step=None,
                last_layer=None, cond=None):
        true = true.contiguous()
        pred = pred.contiguous()

        # now the GAN part
        if optimizer_idx == 0:  # train encoder+decoder+logvar
            nll_loss = self.loss_nll(true, pred)
            kl_loss = self.loss_kl(mean, log_var)

            loss = nll_loss + kl_loss
            losses = {
                'loss.kl': kl_loss,
                'loss.nll': nll_loss,
            }

            if self.use_gan:
                g_loss = self.loss_g(pred, global_step, cond)

                if self.disc_factor > 0.0:
                    try:
                        d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                    except RuntimeError:
                        assert not self.training
                        d_weight = torch.tensor(0.0)
                else:
                    d_weight = torch.tensor(0.0)

                g_loss = d_weight * g_loss
                loss = loss + g_loss
                losses['loss.g'] = g_loss

            losses['loss'] = loss
            return losses

        if optimizer_idx == 1 and self.use_gan:  # train the discriminator
            loss = self.loss_d(true, pred, cond, global_step)
            return {'loss': loss}

    def loss_nll(self, true, pred):
        """Negative Log Likelihood Loss"""
        rec_loss = self.loss_rec(true, pred)
        if self.use_lpips:
            p_loss = self.loss_lpips(true, pred)
            rec_loss = rec_loss + p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        return self.nll_weight * nll_loss.mean()

    def loss_rec(self, true, pred):
        return torch.abs(true - pred)

    def loss_lpips(self, true, pred):
        return self.perceptual_weight * self.perceptual_loss(true, pred)

    def loss_kl(self, mean, log_var):
        return self.kl_weight * self.re_parametrize.loss(mean, log_var)

    def loss_g(self, pred, global_step=0, cond=None):
        # generator update
        if cond is not None:
            pred = torch.cat((pred, cond), dim=1)
        logits_fake = self.discriminator(pred)
        g_loss = -torch.mean(logits_fake)

        disc_factor = self.adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

        return disc_factor * g_loss

    def loss_d(self, true, pred, cond, global_step=0):
        true, pred = true.detach(), pred.detach()
        if cond is not None:
            true = torch.cat((true, cond), dim=1)
            pred = torch.cat((pred, cond), dim=1)

        logits_real = self.discriminator(true)
        logits_fake = self.discriminator(pred)
        disc_factor = self.adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        d_loss = disc_factor * 0.5 * self.disc_loss(logits_real, logits_fake)

        return d_loss

    @staticmethod
    def adopt_weight(weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity
    refer to:
        paper:
            https://arxiv.org/abs/1801.03924
        code:
            https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
        pretrain model:
            https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1
    """

    def __init__(self, chns=(64, 128, 256, 512, 512), drop_prob=0.5):
        super().__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

        self.net = Vgg16(pretrained=True, requires_grad=False)
        self.lins = nn.ModuleList([nn.Sequential(
            nn.Dropout(drop_prob),
            nn.Conv2d(ch, 1, 1, stride=1, padding=0, bias=False)
        ) for ch in chns])
        self.requires_grad_(False)

    def forward(self, true, pred):
        in0_input, in1_input = (self.scale_layer(true), self.scale_layer(pred))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        res = []
        for i, lin in enumerate(self.lins):
            feat0, feat1 = self.normalize_tensor(outs0[i]), self.normalize_tensor(outs1[i])
            diff = (feat0 - feat1) ** 2
            res.append(lin(diff).mean([2, 3], keepdim=True))

        val = res[0]
        for r in res[1:]:
            val += r
        return val

    def scale_layer(self, inp):
        return (inp - self.shift) / self.scale

    @staticmethod
    def normalize_tensor(x, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + eps)


class Vgg16(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super().__init__()
        from torchvision import models
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features

        self.slices = nn.ModuleList([
            nn.Sequential(*[vgg_pretrained_features[x] for x in range(4)]),
            nn.Sequential(*[vgg_pretrained_features[x] for x in range(4, 9)]),
            nn.Sequential(*[vgg_pretrained_features[x] for x in range(9, 16)]),
            nn.Sequential(*[vgg_pretrained_features[x] for x in range(16, 23)]),
            nn.Sequential(*[vgg_pretrained_features[x] for x in range(23, 30)])
        ])

        self.requires_grad_(requires_grad)

    def forward(self, x):
        hidden = []
        h = x
        for m in self.slices:
            h = m(h)
            hidden.append(h)

        return hidden


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False,
                 allow_reverse_init=False):
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, x):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            mean = flatten.mean(1)[None, :, None, None]
            std = flatten.std(1)[None, :, None, None]
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, x, reverse=False):
        if reverse:
            return self.reverse(x)
        else:
            return self.normal(x)

    def normal(self, x):
        if len(x.shape) == 2:
            x = x[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        b, _, h, w = x.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(x)
            self.initialized.fill_(1)

        x = self.scale * (x + self.loc)

        if squeeze:
            x = x.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = h * w * torch.sum(log_abs)
            logdet = logdet * torch.ones(b).to(x)
            return x, logdet

        return x

    def reverse(self, x):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(x)
                self.initialized.fill_(1)

        if len(x.shape) == 2:
            x = x[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        x = x / self.scale - self.loc

        if squeeze:
            x = x.squeeze(-1).squeeze(-1)
        return x
