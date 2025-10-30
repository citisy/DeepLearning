import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from torch import nn

from .ddpm import extract
from .. import bundles
from utils import op_utils

make_schedule_fn = op_utils.RegisterTables()
make_scaling_fn = op_utils.RegisterTables()


class Config(bundles.Config):
    schedule_config = dict()

    scaling_config = dict()

    @classmethod
    def make_full_config(cls) -> dict:
        return {
            '': dict(
                schedule='LegacyDDPMSchedule',
                scaling='EpsScaling',
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
        sigmas = self.make_sigmas()
        st = self.make_st()

        self.register_buffer('sigmas', sigmas * st, persistent=False)

    def _apply(self, fn, recurse=True):
        """apply for meta load"""
        if self.sigmas.is_meta:
            sigmas = self.make_sigmas()
            st = self.make_st()
            self.register_buffer('sigmas', sigmas * st)
        return super()._apply(fn, recurse)

    def make_timesteps(self, i0=None, num_steps=None):
        num_steps = num_steps or self.num_steps
        # note, must start with self.timesteps
        timestep_seq = np.linspace(self.timesteps, 0, num_steps, endpoint=False).astype(int)[::-1]
        if i0:
            timestep_seq = timestep_seq[:i0]
        return timestep_seq

    def make_sigmas(self):
        raise NotImplementedError

    def make_st(self):
        return torch.ones(1, dtype=torch.long)


@make_schedule_fn.add_register()
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


@make_schedule_fn.add_register()
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


@make_schedule_fn.add_register()
class ExponentialSchedule(Schedule):
    sigma_min = 0.002
    sigma_max = 80.0

    def make_sigmas(self):
        sigmas = torch.linspace(math.log(self.sigma_max), math.log(self.sigma_min), self.timesteps).exp()
        return sigmas


@make_schedule_fn.add_register()
class PolyExponentialSchedule(Schedule):
    sigma_min = 0.002
    sigma_max = 80.0
    rho = 1.0

    def make_sigmas(self):
        ramp = torch.linspace(1, 0, self.timesteps) ** self.rho
        sigmas = torch.exp(ramp * (math.log(self.sigma_max) - math.log(self.sigma_min)) + math.log(self.sigma_min))
        return sigmas


@make_schedule_fn.add_register()
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

    def make_c_skip(self, sigma):
        raise NotImplementedError

    def make_c_out(self, sigma):
        raise NotImplementedError

    def make_c_in(self, sigma):
        return 1 / (sigma ** 2 + 1.0) ** 0.5

    def make_c_noise(self, sigma):
        return sigma.clone()

    def predict_real(self, x_0, t, noise):
        raise NotImplementedError


@make_scaling_fn.add_register()
class EpsScaling(Scaling):
    def make_c_skip(self, sigma):
        return torch.ones_like(sigma, device=sigma.device)

    def make_c_out(self, sigma):
        return -sigma

    def predict_real(self, x_0, t, noise):
        return noise


@make_scaling_fn.add_register()
class VScaling(Scaling):
    def make_c_skip(self, sigma):
        return 1.0 / (sigma ** 2 + 1.0)

    def make_c_out(self, sigma):
        return -sigma / (sigma ** 2 + 1.0) ** 0.5

    def predict_real(self, x_0, t, noise):
        return noise


@make_scaling_fn.add_register()
class EDMEpsScaling(Scaling):
    sigma_data = 1.

    def make_c_skip(self, sigma):
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def make_c_out(self, sigma):
        return sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5

    def make_c_noise(self, sigma):
        return 0.25 * sigma.log()

    def predict_real(self, x_0, t, noise):
        return noise


@make_scaling_fn.add_register()
class EDMVScaling(Scaling):
    def make_c_skip(self, sigma):
        return 1.0 / (sigma ** 2 + 1.0)

    def make_c_out(self, sigma):
        return -sigma / (sigma ** 2 + 1.0) ** 0.5

    def make_c_noise(self, sigma):
        return 0.25 * sigma.log()


class Sampler(nn.Module):
    self_condition = False

    def __init__(self, schedule: Schedule | str, scaling: Scaling | str,
                 schedule_config=dict(), scaling_config=dict(),
                 **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.schedule = make_schedule_fn.get(schedule, schedule)(**schedule_config)
        self.scaling = make_scaling_fn.get(scaling, scaling)(**scaling_config)

    @property
    def num_steps(self):
        return self.schedule.num_steps

    def loss(self, diffuse_func, x_0, noise=None, **kwargs):
        raise NotImplementedError

    def q_sample(self, x0, sigma, noise=None):
        raise NotImplementedError

    def forward(self, diffuse_func, x_t, i0=None, callback_fn=None, num_steps=None, **p_sample_kwargs):
        timestep_seq = self.schedule.make_timesteps(i0, num_steps=num_steps)
        num_steps = num_steps or len(timestep_seq)
        # previous sequence
        timestep_prev_seq = np.append(np.array([0]), timestep_seq[:-1])
        x_0 = None
        if callback_fn:
            callback_fn(x_t, self.schedule.timesteps)

        for i in reversed(range(len(timestep_seq))):
            self_cond = x_0 if self.self_condition else None
            x_t, x_0 = self.p_sample(diffuse_func, x_t, timestep_seq[i], prev_t=timestep_prev_seq[i], x_self_cond=self_cond, num_steps=num_steps, **p_sample_kwargs)

            if callback_fn:
                callback_fn(x_t, timestep_seq[i])
        return x_t

    def p_sample(self, diffuse_func, x_t, t, **diffuse_kwargs):
        raise NotImplementedError

    def scale_x_t(self, x_t):
        raise NotImplementedError

    def make_timesteps(self, i0=None, num_steps=None):
        return self.schedule.make_timesteps(i0, num_steps=num_steps)


class EulerSampler(Sampler):
    # for p_sample
    s_tmin = 0.0
    s_tmax = 999.0
    s_churn = 0.0

    def loss(self, diffuse_func, x_0, noise=None, **kwargs):
        b, c, h, w = x_0.shape
        t = torch.randint(0, self.schedule.timesteps, (b,), device=x_0.device).long()
        if noise is None:
            noise = torch.randn_like(x_0)

        sigma = extract(self.schedule.sigmas, t, x_0.shape)

        c_skip, c_out, c_in, c_noise = self.scaling(sigma)
        x_t = self.q_sample(x_0, t, noise=noise)
        pred = diffuse_func(x_t * c_in, self.sigma_to_idx(sigma), **kwargs)
        real = self.scaling.predict_real(x_0, t, noise)

        loss = F.mse_loss(pred.float(), real.float(), reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')
        return loss.mean()

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)

        return self.predict_x_t(x0, t, noise)

    def predict_x_t(self, x_0, t, noise):
        # x_t = x_0 + s_t * z_t
        return x_0 + noise * extract(self.schedule.sigmas, t, x_0.shape)

    def scale_x_t(self, x_t):
        return x_t * torch.sqrt(1.0 + self.schedule.sigmas[-1] ** 2.0)

    def p_sample(self, diffuse_func, x_t, t, prev_t=None, num_steps=None, **diffuse_kwargs):
        # todo: add more sample methods
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_t, device=x_t.device, dtype=torch.long)

        sigma = extract(self.schedule.sigmas, t, x_t.shape)
        next_sigma = extract(self.schedule.sigmas, prev_t, x_t.shape)

        gamma = torch.where(
            torch.logical_and(self.s_tmin <= sigma, sigma <= self.s_tmax),
            min(self.s_churn / (num_steps - 1 + 1e-8), 2 ** 0.5 - 1),
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

        d = diffuse_func(c_in * x_t, possible_t, **diffuse_kwargs) * c_out + x_t * c_skip

        d = (x_t - d) / sigma_hat
        dt = next_sigma - sigma_hat

        x_t = x_t + d * dt
        return x_t, None

    def sigma_to_idx(self, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma.reshape(sigma.shape[0])
        dists = sigma - self.schedule.sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape)
