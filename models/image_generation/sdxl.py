import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
from . import ldm, ddpm, ddim, sdv1, sdv2
from .ddpm import extract


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

    # for sigmas schedule
    LEGACY_DDPM = 1
    EDM = 2

    # for sampler scaling
    Z = 2  # same to PRED_Z in ddim
    V = 3  # same to PRED_V in ddim
    EDM_Z = 4
    EDM_V = 5

    xl_model = dict(
        scale=5,
        scale_factor=0.13025,
        objective=Z,
    )

    legacy_v2_embedder = dict(
        target='models.image_generation.sdv2.OpenCLIPEmbedder',
        input_key=TXT,
        params=dict()
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
        adm_in_channels=2816,  # 1028 * 2 + 256 * 3
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
        adm_in_channels=2560,  # 1028 * 2 + 256 * 2
        unit_dim=384,
        ch_mult=(1, 2, 4, 4),
        attend_layers=(1, 2),
        transformer_depth=4,
    )

    default_model = 'xl_base'

    @classmethod
    def make_full_config(cls):
        config_dict = dict(
            # todo: legacy_v2 supported

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


class Model(ldm.Model):
    """https://github.com/Stability-AI/generative-models"""

    num_steps = 40
    schedule_type = Config.LEGACY_DDPM

    # for edm schedule
    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0

    # for p_sample
    s_tmin = 0.0
    s_tmax = 999.0
    s_churn = 0.0

    # for EDM_Z
    sigma_data = 0.5

    # for video
    num_frames = 14

    def make_schedule(self):
        if self.schedule_type == Config.EDM:  # edm
            ramp = torch.linspace(0, 1, self.timesteps)
            min_inv_rho = self.sigma_min ** (1 / self.rho)
            max_inv_rho = self.sigma_max ** (1 / self.rho)
            sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho

        elif self.schedule_type == Config.LEGACY_DDPM:  # legacy ddpm
            from .ddpm import linear_beta_schedule
            betas = linear_beta_schedule(self.timesteps, start=0.00085 ** 0.5, end=0.0120 ** 0.5) ** 2
            betas = betas.to(torch.float32)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
            sigmas = torch.cat([sigmas.new_zeros([1]), sigmas])

        else:
            raise ValueError(f'unknown objective {self.schedule_type}')

        st = torch.ones(self.timesteps + 1, dtype=torch.long)
        self.register_buffer('sigmas', sigmas)
        self.register_buffer('st', st)

    def make_cond(self, cond_config=[], **kwargs):
        return EmbedderWarp(cond_config)

    def p_sample_loop(self, x_t, **kwargs):
        x_t *= torch.sqrt(1.0 + self.sigmas[-1] ** 2.0)
        return super().p_sample_loop(x_t, **kwargs)

    def make_timesteps(self, t0=None):
        # note, must start with self.timesteps
        timestep_seq = np.linspace(self.timesteps, 0, self.num_steps, endpoint=False).astype(int)[::-1]
        if t0:
            timestep_seq = timestep_seq[:t0]
        return timestep_seq

    def p_sample(self, x_t, t, prev_t=None, x_self_cond=None, **kwargs):
        # todo: add more sample methods
        s_in = self.st[t]
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_t, device=x_t.device, dtype=torch.long)

        sigma = extract(self.sigmas, t, x_t.shape) * s_in
        next_sigma = extract(self.sigmas, prev_t, x_t.shape) * s_in

        gamma = (
            min(self.s_churn / (self.num_steps - 1), 2 ** 0.5 - 1)
            if self.s_tmin <= sigma <= self.s_tmax
            else 0.0
        )

        sigma_hat = sigma * (gamma + 1.0)

        if gamma > 0:
            eps = torch.randn_like(x_t) * self.s_noise
            x_t = x_t + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5

        possible_sigma = self.sigmas[self.sigma_to_idx(sigma_hat)]
        c_skip, c_out, c_in, c_noise = self.make_scaling(possible_sigma)
        possible_t = self.sigma_to_idx(c_noise)
        possible_t = possible_t - 1

        d = self.diffuse(c_in * x_t, possible_t, **kwargs) * c_out + x_t * c_skip

        d = (x_t - d) / sigma_hat
        dt = next_sigma - sigma_hat

        x_t = x_t + d * dt
        return x_t, None

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(sigma.shape[0])
        dists = sigma - self.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)

    def make_scaling(self, sigma):
        if self.objective == Config.Z:
            c_skip = torch.ones_like(sigma, device=sigma.device)
            c_out = -sigma
            c_in = 1 / (sigma ** 2 + 1.0) ** 0.5
            c_noise = sigma.clone()

        elif self.objective == Config.V:
            c_skip = 1.0 / (sigma ** 2 + 1.0)
            c_out = -sigma / (sigma ** 2 + 1.0) ** 0.5
            c_in = 1.0 / (sigma ** 2 + 1.0) ** 0.5
            c_noise = sigma.clone()

        elif self.objective == Config.EDM_Z:
            c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
            c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
            c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
            c_noise = 0.25 * sigma.log()

        elif self.objective == Config.EDM_V:
            c_skip = 1.0 / (sigma ** 2 + 1.0)
            c_out = -sigma / (sigma ** 2 + 1.0) ** 0.5
            c_in = 1.0 / (sigma ** 2 + 1.0) ** 0.5
            c_noise = 0.25 * sigma.log()

        else:
            raise ValueError(f'unknown objective {self.objective}')

        return c_skip, c_out, c_in, c_noise

    def make_txt_cond(self, text, neg_text=None, **kwargs) -> dict:
        if not neg_text:
            neg_text = [''] * len(text)

        default_value = {
            Config.ORIGINAL_SIZE_AS_TUPLE: self.image_size,
            Config.CROP_COORDS_TOP_LEFT: (0, 0),
            Config.TARGET_SIZE_AS_TUPLE: self.image_size,
        }

        value_dicts = []
        for prompt, negative_prompt in zip(text, neg_text):
            value_dict = {}
            for k in self.cond.input_keys:
                if k == Config.TXT:
                    value_dict.update(
                        prompt=prompt,
                        negative_prompt=negative_prompt
                    )
                else:
                    value_dict[k] = kwargs.get(k, default_value[k])

            value_dicts.append(value_dict)

        c_values, uc_values = self.cond(value_dicts, self.scale > 1)

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

            if isinstance(self.scale, list):
                scale = torch.tensor(self.scale).to(e_t_uncond.device)
                scale = repeat(scale, "1 t -> b t", b=e_t_uncond.shape[0])
                scale = scale.view((*scale.shape, *e_t.shape[2:]))
                e_t_uncond = rearrange(e_t_uncond, "(b t) ... -> b t ...", t=self.num_frames)
                e_t = rearrange(e_t, "(b t) ... -> b t ...", t=self.num_frames)
                e_t = rearrange(e_t_uncond + scale * (e_t - e_t_uncond), "b t ... -> (b t) ...")
            else:
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

    def forward(self, value_dicts, return_uc=True):
        batch, batch_uc = self.get_batch(value_dicts)
        c_values = self.get_cond(batch)
        # todo: why filter txt?
        uc_values = self.get_cond(batch_uc, [Config.TXT]) if return_uc else None
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
