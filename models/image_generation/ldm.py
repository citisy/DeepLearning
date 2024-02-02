import functools
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from utils import torch_utils
from .ddim import Model as DmModel, Config as DmConfig
from .ddpm import RandomOrLearnedSinusoidalPosEmb, SinusoidalPosEmb, ResnetBlock, make_norm
from .VAE import Model as VaeModel, Config as VaeConfig
from ..layers import Linear, Conv, Upsample, Downsample
from ..attentions import CrossAttention2D


class Config:
    in_module = dict(
        pretrain_model='openai/clip-vit-large-patch14'
    )
    model = dict(
        ddim_timesteps=50,
        timesteps=1000,
        ddim_eta=0.,
        scale=7.5,
        scale_factor=0.18215,
        strength=0.75,  # for img2img
        objective=DmConfig.PRED_Z
    )

    backbone = dict(
        ch_mult=[1, 2, 4, 4],
        context_dim=768,
        use_checkpoint=True,
    )
    head = dict(
        img_ch=3,
        **VaeConfig.get('backbone_32x32x4')
    )

    @classmethod
    def get(cls, name=None):
        return dict(
            in_module_config=cls.in_module,
            model_config=cls.model,
            backbone_config=cls.backbone,
            head_config=cls.head
        )


def convert_ol_weights(state_dict):
    """convert weights from official model to my own model
    https://github.com/CompVis/latent-diffusion?tab=readme-ov-file#model-zoo

    Usage:
        .. code-block:: python

            state_dict = torch.load(self.pretrain_model, map_location=self.device)['state_dict']
            state_dict = convert_hf_weights(state_dict)
            Model(...).load_state_dict(state_dict)
    """

    convert_dict = {
        'first_stage_model': 'head',
        'first_stage_model.{0}.block.{1}.norm{2}.': 'head.{0}.blocks.{1}.fn.conv{2}.norm.',
        'first_stage_model.{0}.block.{1}.conv{2}.': 'head.{0}.blocks.{1}.fn.conv{2}.conv.',
        'first_stage_model.{0}.block.{1}.nin_shortcut': 'head.{0}.blocks.{1}.project_fn',
        'first_stage_model.{0}sample.conv': 'head.{0}sample.fn.1',
        'first_stage_model.{0}.mid.block_{1}.norm{2}.': 'head.{0}.neck.block_{1}.fn.conv{2}.norm.',
        'first_stage_model.{0}.mid.block_{1}.conv{2}.': 'head.{0}.neck.block_{1}.fn.conv{2}.conv.',
        'first_stage_model.{0}.mid.attn_1.norm': 'head.{0}.neck.attn.0',
        'first_stage_model.{0}.mid.attn_1.q': 'head.{0}.neck.attn.1.to_qkv.0',
        'first_stage_model.{0}.mid.attn_1.k': 'head.{0}.neck.attn.1.to_qkv.1',
        'first_stage_model.{0}.mid.attn_1.v': 'head.{0}.neck.attn.1.to_qkv.2',
        'first_stage_model.{0}.mid.attn_1.proj_out': 'head.{0}.neck.attn.1.to_out',
        'first_stage_model.{0}.norm_out': 'head.{0}.head.norm',
        'first_stage_model.{0}.conv_out': 'head.{0}.head.conv',

        'model.diffusion_model': 'backbone',
        'model.diffusion_model.time_embed.0': 'backbone.time_embed.1.linear',
        'model.diffusion_model.time_embed.2': 'backbone.time_embed.2.linear',
        'model.diffusion_model.{2}.0.0': 'backbone.{2}.0.layers.0',
        'model.diffusion_model.{0}.0.in_layers.0': 'backbone.{0}.layers.0.in_layers.norm',
        'model.diffusion_model.{0}.0.in_layers.2': 'backbone.{0}.layers.0.in_layers.conv',
        'model.diffusion_model.{0}.0.emb_layers.1': 'backbone.{0}.layers.0.emb_layers.linear',
        'model.diffusion_model.{0}.0.out_layers.0': 'backbone.{0}.layers.0.norm',
        'model.diffusion_model.{0}.0.out_layers.3': 'backbone.{0}.layers.0.out_layers.conv',
        'model.diffusion_model.{0}.1.norm': 'backbone.{0}.layers.1.proj_in.norm',
        'model.diffusion_model.{0}.1.proj_in': 'backbone.{0}.layers.1.proj_in.conv',
        'model.diffusion_model.{0}.1.transformer_blocks.0.attn{1}.to_q': 'backbone.{0}.layers.1.transformer_blocks.0.attn{1}.to_qkv.0',
        'model.diffusion_model.{0}.1.transformer_blocks.0.attn{1}.to_k': 'backbone.{0}.layers.1.transformer_blocks.0.attn{1}.to_qkv.1',
        'model.diffusion_model.{0}.1.transformer_blocks.0.attn{1}.to_v': 'backbone.{0}.layers.1.transformer_blocks.0.attn{1}.to_qkv.2',
        'model.diffusion_model.{0}.1.transformer_blocks.0.attn{1}.to_out.0': 'backbone.{0}.layers.1.transformer_blocks.0.attn{1}.to_out.linear',
        'model.diffusion_model.{0}.1.transformer_blocks.0.ff.net.0.proj': 'backbone.{0}.layers.1.transformer_blocks.0.ff.0.proj',
        'model.diffusion_model.{0}.1.transformer_blocks.0.ff.net.2': 'backbone.{0}.layers.1.transformer_blocks.0.ff.2',
        'model.diffusion_model.{0}.1.transformer_blocks.0.norm{1}.': 'backbone.{0}.layers.1.transformer_blocks.0.norm{1}.',
        'model.diffusion_model.{0}.1.proj_out': 'backbone.{0}.layers.1.proj_out',
        'model.diffusion_model.{0}.1.conv': 'backbone.{0}.layers.1.op.1',
        'model.diffusion_model.{0}.2.conv': 'backbone.{0}.layers.2.op.1',
        'model.diffusion_model.{0}.0.op': 'backbone.{0}.layers.0.op',
        'model.diffusion_model.{0}.0.skip_connection': 'backbone.{0}.layers.0.proj',
        'model.diffusion_model.middle_block.2.in_layers.0': 'backbone.middle_block.layers.2.in_layers.norm',
        'model.diffusion_model.middle_block.2.in_layers.2': 'backbone.middle_block.layers.2.in_layers.conv',
        'model.diffusion_model.middle_block.2.out_layers.0': 'backbone.middle_block.layers.2.norm',
        'model.diffusion_model.middle_block.2.emb_layers.1': 'backbone.middle_block.layers.2.emb_layers.linear',
        'model.diffusion_model.middle_block.2.out_layers.3': 'backbone.middle_block.layers.2.out_layers.conv',
        'model.diffusion_model.out.0': 'backbone.out.norm',
        'model.diffusion_model.out.2': 'backbone.out.conv',

        'cond_stage_model': 'in_module'

    }
    state_dict = torch_utils.convert_state_dict(state_dict, convert_dict)

    return state_dict


class Model(DmModel):
    """refer to:
    paper:
        - High-Resolution Image Synthesis with Latent Diffusion Models
    code:
        - https://github.com/CompVis/latent-diffusion
        - https://github.com/CompVis/stable-diffusion
    """

    def __init__(self, image_size,
                 in_module_trainable=False, in_module=None,
                 in_module_config=Config.in_module, model_config=Config.model,
                 backbone_config=Config.backbone, head_config=Config.head):
        if in_module is None:
            in_module = CLIPEmbedder(**in_module_config)

        backbone = UNetModel(**backbone_config)
        head = VaeModel(**head_config)

        if not hasattr(in_module, 'encode'):
            in_module.encode = in_module.__call__
        if not hasattr(head, 'decode'):
            head.decode = head.__call__

        if not in_module_trainable:
            torch_utils.freeze_layers(in_module)
        torch_utils.freeze_layers(head)

        self.register_model_config(**model_config)

        super().__init__(
            4,  # inner ch for backbone
            image_size // head.encoder.down_scale,  # inner size for backbone
            in_module=in_module,
            backbone=backbone,
            head=head,
            **model_config
        )

    def register_model_config(self, scale=1., scale_factor=1.0, strength=0.75, **kwargs):
        self.scale = scale
        self.scale_factor = scale_factor
        self.strength = strength

    def post_process(self, x=None, text=None, image=None, **kwargs):
        b = len(text)

        c = self.in_module.encode(text)
        uc = None
        if self.scale != 1.0:
            uc = self.in_module.encode([''] * b)

        # make x_t
        if x is None:  # txt2img
            x = self.gen_x_t(b, c.device)
            t0 = None

        else:  # img2img
            x, t0 = self.make_image_cond(x)

        z = super().post_process(x, t0=t0, cond=c, un_cond=uc)
        z = 1. / self.scale_factor * z
        images = self.head.decode(z)

        return images

    def make_image_cond(self, image):
        image = 2. * image - 1.
        z, _, _ = self.head.encode(image)
        x0 = self.scale_factor * z

        ddim_timestep_seq = self.make_ddim_timesteps()
        t0 = int(self.strength * self.ddim_timesteps)
        t = ddim_timestep_seq[t0]
        t = torch.full((x0.shape[0],), t, device=x0.device, dtype=torch.long)
        noise = torch.randn_like(x0)

        x = self.predict_x_t(x0, t, noise)
        return x, t0

    def diffuse(self, x, time, cond=None, un_cond=None, **kwargs):
        if un_cond is not None:
            x = torch.cat([x] * 2)
            time = torch.cat([time] * 2)
            cond = torch.cat([un_cond, cond])

        z = self.backbone(x, timesteps=time, context=cond)
        if un_cond is None:
            e_t = z
        else:
            e_t_uncond, e_t = z.chunk(2)
            e_t = e_t_uncond + self.scale * (e_t - e_t_uncond)

        return e_t


class CLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from Hugging Face)
    see https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K"""

    def __init__(self, pretrain_model=None, max_length=77, load_weight=False):
        super().__init__()
        from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextConfig

        pretrain_model = pretrain_model or 'openai/clip-vit-large-patch14'
        self.tokenizer = CLIPTokenizer.from_pretrained(pretrain_model)

        if load_weight:
            self.transformer = CLIPTextModel.from_pretrained(pretrain_model)

        else:
            configuration = CLIPTextConfig.from_pretrained(pretrain_model)
            self.transformer = CLIPTextModel(configuration)
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.transformer.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z


class UNetModel(nn.Module):
    """base on Unet, add attention, res, etc."""

    def __init__(
            self,
            in_ch=4,
            unit_dim=320,
            out_ch=4,
            num_res_blocks=2,
            attend_layers=(0, 1, 2),
            groups=32,
            ch_mult=(1, 2, 4, 8),
            conv_resample=True,
            num_classes=None,
            use_fp16=False,
            use_checkpoint=False,
            num_heads=8,
            context_dim=768,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            sinusoidal_pos_emb_theta=10000,
            learned_sinusoidal_cond=False,
            random_fourier_features=False,
            learned_sinusoidal_dim=16,
    ):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.num_classes = num_classes
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.predict_codebook_ids = n_embed is not None

        time_emb_dim = unit_dim * 4

        # helper
        make_res = functools.partial(ResnetBlock, groups=groups, use_checkpoint=use_checkpoint, time_emb_dim=time_emb_dim)
        make_trans = functools.partial(TransformerBlock, context_dim=context_dim, use_checkpoint=use_checkpoint)

        if learned_sinusoidal_cond:
            sin_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sin_pos_emb = SinusoidalPosEmb(unit_dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = unit_dim

        self.time_embed = nn.Sequential(
            sin_pos_emb,
            Linear(fourier_dim, time_emb_dim, mode='la', act=nn.SiLU()),
            Linear(time_emb_dim, time_emb_dim, mode='l'),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_emb_dim)

        out_ch = unit_dim
        layers = [TimestepEmbedSequential(nn.Conv2d(in_ch, out_ch, 3, padding=1))]
        input_block_chans = [out_ch]

        num_stages = len(ch_mult)
        in_ch = out_ch
        for i, mult in enumerate(ch_mult):
            is_bottom = i == num_stages - 1

            for _ in range(num_res_blocks):
                out_ch = mult * unit_dim
                blocks = [make_res(in_ch, out_ch)]
                if i in attend_layers:
                    head_dim = out_ch // num_heads
                    blocks.append(make_trans(out_ch, num_heads, head_dim))

                layers.append(TimestepEmbedSequential(*blocks))
                input_block_chans.append(out_ch)
                in_ch = out_ch
            if not is_bottom:
                layers.append(TimestepEmbedSequential(
                    Downsample(out_ch, out_ch, use_conv=conv_resample)
                ))
                input_block_chans.append(out_ch)

        self.input_blocks = nn.ModuleList(layers)

        head_dim = out_ch // num_heads
        self.middle_block = TimestepEmbedSequential(
            make_res(out_ch, out_ch),
            make_trans(out_ch, num_heads, head_dim),
            make_res(out_ch, out_ch),
        )

        layers = []
        for i, mult in list(enumerate(ch_mult))[::-1]:
            is_top = i == 0
            for j in range(num_res_blocks + 1):
                is_block_bottom = j == num_res_blocks

                ich = input_block_chans.pop()
                out_ch = unit_dim * mult
                blocks = [make_res(in_ch + ich, out_ch)]
                if i in attend_layers:
                    head_dim = out_ch // num_heads
                    blocks.append(make_trans(out_ch, num_heads, head_dim))

                if not is_top and is_block_bottom:
                    blocks.append(Upsample(out_ch, out_ch, use_conv=conv_resample))

                layers.append(TimestepEmbedSequential(*blocks))
                in_ch = out_ch

        self.output_blocks = nn.ModuleList(layers)

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                make_norm(groups, in_ch),
                nn.Conv2d(in_ch, n_embed, 1),
            )

        else:
            self.out = Conv(in_ch, self.out_channels, 3, mode='nac', act=nn.SiLU(), norm=make_norm(groups, unit_dim))

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        emb = self.time_embed(timesteps)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        hs = []
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class TimestepEmbedSequential(nn.Module):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, emb=None, context=None):
        for layer in self.layers:
            if isinstance(layer, ResnetBlock):
                x = layer(x, emb)
            elif isinstance(layer, TransformerBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, in_ch, n_heads, head_dim, groups=32,
                 depth=1, dropout=0., context_dim=None, use_checkpoint=False):
        super().__init__()
        self.in_channels = in_ch
        model_dim = n_heads * head_dim
        self.proj_in = Conv(in_ch, model_dim, 1, mode='nc', norm=make_norm(groups, in_ch, eps=1e-6))  # note, original code use `eps=1e-6`

        self.transformer_blocks = nn.ModuleList([BasicTransformerBlock(
            model_dim, n_heads, head_dim, drop_prob=dropout, context_dim=context_dim, use_checkpoint=use_checkpoint
        ) for _ in range(depth)])

        self.proj_out = nn.Conv2d(model_dim, in_ch, 1, stride=1, padding=0)

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        y = self.proj_in(x)
        y = rearrange(y, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            y = block(y, context=context)
        y = rearrange(y, 'b (h w) c -> b c h w', h=h, w=w)
        y = self.proj_out(y)
        return y + x


class BasicTransformerBlock(nn.Module):
    def __init__(self, query_dim, n_heads, head_dim, drop_prob=0., context_dim=None, gated_ff=True, use_checkpoint=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(query_dim)
        self.attn1 = CrossAttention2D(query_dim=query_dim, n_heads=n_heads, head_dim=head_dim, drop_prob=drop_prob, bias=False)  # is a self-attention

        self.norm2 = nn.LayerNorm(query_dim)
        self.attn2 = CrossAttention2D(query_dim=query_dim, context_dim=context_dim, n_heads=n_heads, head_dim=head_dim, drop_prob=drop_prob, bias=False)  # is self-attn if context is none

        self.norm3 = nn.LayerNorm(query_dim)
        self.ff = FeedForward(query_dim, drop_prob=drop_prob, glu=gated_ff)

        self.use_checkpoint = use_checkpoint

    def forward(self, x, context=None):
        return torch_utils.checkpoint(self._forward, (x, context), self.parameters(), self.use_checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class FeedForward(nn.Sequential):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, drop_prob=0.):
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        super().__init__(
            project_in,
            nn.Dropout(drop_prob),
            nn.Linear(inner_dim, dim_out)
        )


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
