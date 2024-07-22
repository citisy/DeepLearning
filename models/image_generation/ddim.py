import torch
import numpy as np
from .ddpm import Model as Model_, extract, Config, Sampler as Sampler_


class Model(Model_):
    """refer to:
    paper:
        - DENOISING DIFFUSION IMPLICIT MODELS
    code:
        - https://github.com/lucidrains/denoising-diffusion-pytorch
    """

    def make_sampler(self, sampler_config=Config.sampler_config, **kwargs):
        self.sampler = Sampler(**sampler_config)


class Sampler(Sampler_):
    ddim_eta = 0.
    num_steps = 50
    ddim_discr_method = 'uniform'

    def make_timesteps(self, t0=None):
        if self.ddim_discr_method == 'uniform':
            c = self.timesteps // self.num_steps
            timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif self.ddim_discr_method == 'quad':
            timestep_seq = ((np.linspace(0, np.sqrt(self.timesteps * .8), self.num_steps)) ** 2).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{self.ddim_discr_method}"')

        if t0:
            timestep_seq = timestep_seq[:t0]

        # note, add one to get the final alpha values right (the ones from first scale to data during sampling)
        return timestep_seq + 1

    def forward(self, diffuse_func, x_t, t0=None, callback_fn=None, **kwargs):
        timestep_seq = self.make_timesteps(t0)
        # previous sequence
        timestep_prev_seq = np.append(np.array([0]), timestep_seq[:-1])
        x_0 = None
        if callback_fn:
            callback_fn(x_t, self.timesteps)

        for i in reversed(range(len(timestep_seq))):
            self_cond = x_0 if self.self_condition else None
            x_t, x_0 = self.p_sample(diffuse_func, x_t, timestep_seq[i], timestep_prev_seq[i], self_cond, **kwargs)
            if callback_fn:
                callback_fn(x_t, timestep_seq[i])
        return x_t

    def p_sample(self, diffuse_func, x_t, t, prev_t=None, x_self_cond=None, **kwargs):
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_t, device=x_t.device, dtype=torch.long)

        x_0, pred_noise = self.model_predictions(diffuse_func, x_t, t, x_self_cond, return_pred_noise=True, **kwargs)

        # s_t = \eta * \sqrt{(1−ca_{t−1})/(1−ca_t)} * \sqrt{1−ca_t/ca_{t−1}}
        alpha_cumprod_t = extract(self.alphas_cumprod, t, x_t.shape)
        alpha_cumprod_t_prev = extract(self.alphas_cumprod, prev_t, x_t.shape)
        sigmas_t = self.ddim_eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

        # compute "direction pointing to x_t" of formula (12)
        # x_t = \sqrt{1-ca_{t-1}-s_t^2} * z_t
        dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise

        # x_{t-1} = x_0 * \sqrt{ca_{t−1}} + x_t + z_t * s_t
        x_t_1 = torch.sqrt(alpha_cumprod_t_prev) * x_0 + dir_xt + sigmas_t * torch.randn_like(x_t)

        return x_t_1, x_0
