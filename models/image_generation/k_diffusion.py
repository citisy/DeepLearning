import torch
from torch import nn, einsum
import numpy as np
import math
from .ddpm import extract, append_dims
from .. import bundles


class Config(bundles.Config):
    LEGACY_DDPM = 'LegacyDDPM'
    EDM = 'Karras'
    KARRAS = 'Karras'
    EXPONENTIAL = 'Exponential'
    POLY_EXPONENTIAL = 'PolyExponential'
    VP = 'VP'

    PRED_Z = 2
    PRED_V = 3

    PRED_EDM_Z = 4
    PRED_EDM_V = 5

    schedule_config = dict()

    scaling_config = dict()

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            '': dict(
                schedule=cls.LEGACY_DDPM,
                scaling=cls.PRED_Z,
                schedule_config=cls.schedule_config,
                scaling_config=cls.scaling_config
            )
        }


class Schedule(nn.Module):
    timesteps = 1000
    num_steps = 40

    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.register_buffer('sigmas', self.make_sigmas())

    def make_timesteps(self, t0=None):
        # note, must start with self.timesteps
        timestep_seq = np.linspace(self.timesteps, 0, self.num_steps, endpoint=False).astype(int)[::-1]
        if t0:
            timestep_seq = timestep_seq[:t0]
        return timestep_seq

    def make_sigmas(self):
        raise NotImplementedError


class LegacyDDPMSchedule(Schedule):
    def make_sigmas(self):
        from .ddpm import linear_beta_schedule
        betas = linear_beta_schedule(self.timesteps, start=0.00085 ** 0.5, end=0.0120 ** 0.5) ** 2
        betas = betas.to(torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        sigmas = torch.cat([sigmas.new_zeros([1]), sigmas])

        return sigmas


class EDMSchedule(Schedule):
    """Karras"""
    sigma_min = 0.002
    sigma_max = 80.0
    rho = 7.0

    def make_sigmas(self):
        ramp = torch.linspace(0, 1, self.timesteps)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho

        return sigmas


class ExponentialSchedule(Schedule):
    sigma_min = 0.002
    sigma_max = 80.0

    def make_sigmas(self):
        sigmas = torch.linspace(math.log(self.sigma_max), math.log(self.sigma_min), self.timesteps).exp()
        return sigmas


class PolyExponentialSchedule(Schedule):
    sigma_min = 0.002
    sigma_max = 80.0
    rho = 1.0

    def make_sigmas(self):
        ramp = torch.linspace(1, 0, self.timesteps) ** self.rho
        sigmas = torch.exp(ramp * (math.log(self.sigma_max) - math.log(self.sigma_min)) + math.log(self.sigma_min))
        return sigmas


class VpSchedule(Schedule):
    eps_s = 1e-3
    beta_d = 19.9
    beta_min = 0.1

    def make_sigmas(self):
        t = torch.linspace(1, self.eps_s, self.timesteps)
        sigmas = torch.sqrt(torch.exp(self.beta_d * t ** 2 / 2 + self.beta_min * t) - 1)
        return sigmas


class Scaling(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def forward(self, sigma):
        return (
            self.make_c_skip(sigma),
            self.make_c_out(sigma),
            self.make_c_in(sigma),
            self.make_c_noise(sigma)
        )

    def make_st(self, sigmas, t):
        return torch.ones(1, dtype=torch.long, device=sigmas.device)

    def make_c_skip(self, sigma):
        raise NotImplementedError

    def make_c_out(self, sigma):
        raise NotImplementedError

    def make_c_in(self, sigma):
        return 1 / (sigma ** 2 + 1.0) ** 0.5

    def make_c_noise(self, sigma):
        return sigma.clone()


class EpsScaling(Scaling):
    def make_c_skip(self, sigma):
        return torch.ones_like(sigma, device=sigma.device)

    def make_c_out(self, sigma):
        return -sigma


class VScaling(Scaling):
    def make_c_skip(self, sigma):
        return 1.0 / (sigma ** 2 + 1.0)

    def make_c_out(self, sigma):
        return -sigma / (sigma ** 2 + 1.0) ** 0.5


class EDMEpsScaling(Scaling):
    sigma_data = 1.

    def make_c_skip(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def make_c_out(self, sigma):
        return sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

    def make_c_noise(self, sigma):
        return 0.25 * sigma.log()


class EDMVScaling(Scaling):
    def make_c_skip(self, sigma):
        return 1.0 / (sigma ** 2 + 1.0)

    def make_c_out(self, sigma):
        return -sigma / (sigma ** 2 + 1.0) ** 0.5

    def make_c_noise(self, sigma):
        return 0.25 * sigma.log()


class Sampler(nn.Module):
    schedule_mapping = {
        Config.LEGACY_DDPM: LegacyDDPMSchedule,
        Config.EDM: EDMSchedule,
        Config.EXPONENTIAL: ExponentialSchedule,
        Config.POLY_EXPONENTIAL: PolyExponentialSchedule,
        Config.VP: VpSchedule
    }

    scaling_mapping = {
        Config.PRED_Z: EpsScaling,
        Config.PRED_V: VScaling,
        Config.PRED_EDM_Z: EDMEpsScaling,
        Config.PRED_EDM_V: EDMVScaling
    }

    self_condition = False

    def __init__(self, schedule: Schedule | str, scaling: Scaling | str,
                 schedule_config=dict(), scaling_config=dict(),
                 **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.schedule = self.schedule_mapping.get(schedule, schedule)(**schedule_config)
        self.scaling = self.scaling_mapping.get(scaling, scaling)(**scaling_config)

    def forward(self, diffuse_func, x_t, t0=None, callback_fn=None, **kwargs):
        timestep_seq = self.schedule.make_timesteps(t0)
        # previous sequence
        timestep_prev_seq = np.append(np.array([0]), timestep_seq[:-1])
        x_0 = None
        if callback_fn:
            callback_fn(x_t, self.schedule.timesteps)

        for i in reversed(range(len(timestep_seq))):
            self_cond = x_0 if self.self_condition else None
            x_t, x_0 = self.p_sample(diffuse_func, x_t, timestep_seq[i], prev_t=timestep_prev_seq[i], x_self_cond=self_cond, **kwargs)
            if callback_fn:
                callback_fn(x_t, timestep_seq[i])
        return x_t

    def p_sample(self, diffuse_func, x_t, t, **kwargs):
        raise NotImplementedError


class EulerSampler(Sampler):
    # for p_sample
    s_tmin = 0.0
    s_tmax = 999.0
    s_churn = 0.0

    def loss(self, diffuse_func, x_0, t, noise=None, offset_noise_strength=None, **kwargs):
        c_skip, c_out, c_in = [utils.append_dims(x, input.ndim) for x in self.get_scalings(sigma)]
        noised_input = input + noise * utils.append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, self.sigma_to_t(sigma), **kwargs)
        target = (input - c_skip * noised_input) / c_out
        return (model_output - target).pow(2).flatten(1).mean(1)

    def forward(self, diffuse_func, x_t, **kwargs):
        x_t *= torch.sqrt(1.0 + self.schedule.sigmas[-1] ** 2.0)
        return super().forward(diffuse_func, x_t, **kwargs)

    def p_sample(self, diffuse_func, x_t, t, prev_t=None, **kwargs):
        # todo: add more sample methods
        s_in = self.scaling.make_st(self.schedule.sigmas, t)
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_t, device=x_t.device, dtype=torch.long)

        sigma = extract(self.schedule.sigmas, t, x_t.shape) * s_in
        next_sigma = extract(self.schedule.sigmas, prev_t, x_t.shape) * s_in

        gamma = torch.where(
            torch.logical_and(self.s_tmin <= sigma, sigma <= self.s_tmax),
            min(self.s_churn / (self.schedule.num_steps - 1), 2 ** 0.5 - 1),
            0.
        ).to(sigma)

        sigma_hat = sigma * (gamma + 1.0)

        if torch.any(gamma > 0):
            eps = torch.randn_like(x_t) * self.s_noise
            x_t = x_t + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5

        possible_sigma = self.schedule.sigmas[self.sigma_to_idx(sigma_hat)]
        c_skip, c_out, c_in, c_noise = self.scaling(possible_sigma)
        possible_t = self.sigma_to_idx(c_noise)
        possible_t = possible_t - 1
        c_skip, c_out, c_in, c_noise = c_skip[:, None, None, None], c_out[:, None, None, None], c_in[:, None, None, None], c_noise[:, None, None, None]

        d = diffuse_func(c_in * x_t, possible_t, **kwargs) * c_out + x_t * c_skip

        d = (x_t - d) / sigma_hat
        dt = next_sigma - sigma_hat

        x_t = x_t + d * dt
        return x_t, None

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(sigma.shape[0])
        dists = sigma - self.schedule.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)
