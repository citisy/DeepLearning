import torch
import numpy as np
from .ddpm import Model as Model_, extract


class Model(Model_):
    def __init__(self, ddim_discr_method='uniform', ddim_timesteps=30, ddim_eta=0., **kwargs):
        super().__init__(**kwargs)
        self.ddim_discr_method = ddim_discr_method
        self.ddim_timesteps = ddim_timesteps
        self.ddim_eta = ddim_eta

    def post_process(self, x_t, return_all_timesteps=False):
        if self.ddim_discr_method == 'uniform':
            c = self.timesteps // self.ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.timesteps, c)))
        elif self.ddim_discr_method == 'quad':
            ddim_timestep_seq = ((np.linspace(0, np.sqrt(self.timesteps * .8), self.ddim_timesteps)) ** 2).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{self.ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        x_0 = None
        imgs = [x_t]
        for i in reversed(range(0, self.ddim_timesteps)):
            self_cond = x_0 if self.self_condition else None
            x_t, x_0 = self.p_sample(x_t, ddim_timestep_seq[i], ddim_timestep_prev_seq[i], self_cond)
            imgs.append(x_t)

        ret = x_t if not return_all_timesteps else torch.stack(imgs, dim=1)
        ret = (ret + 1) * 0.5  # unnormalize
        return ret

    def p_sample(self, x_t, t: int, prev_t: int, x_self_cond=None):
        t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        prev_t = torch.full((x_t.shape[0],), prev_t, device=x_t.device, dtype=torch.long)

        pred_noise, x_0 = self.model_predictions(x_t, t, x_self_cond)

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
