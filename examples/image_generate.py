from .base import Process, BaseDataset, WEIGHT
import torch
from torch import nn, optim
from tqdm import tqdm
from utils import os_lib, configs
from data_parse.cv_data_parse.base import DataRegister, DataVisualizer
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply, Apply, channel, RandomChoice
from pathlib import Path
import numpy as np
from datetime import datetime
from metrics import image_generation


class GanOptimizer:
    def __init__(self, optimizer_d, optimizer_g):
        self.optimizer_d = optimizer_d
        self.optimizer_g = optimizer_g

    def state_dict(self):
        return {
            'optimizer_d': self.optimizer_d.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict()
        }

    def load_state_dict(self, dic):
        self.optimizer_d.load_state_dict(dic['optimizer_d'])
        self.optimizer_g.load_state_dict(dic['optimizer_g'])


class IgProcess(Process):
    def on_train_epoch_end(self, total_nums, save_period, val_obj, train_batch_size,
                           losses=None, max_save_weight_num=None, **metric_kwargs):
        if save_period and total_nums % save_period < train_batch_size:
            self.log_info = {'total_nums': total_nums}

            if losses is not None:
                for k, v in losses.items():
                    self.log_info[f'loss/{k}'] = v

            result = self.metric(val_obj, cur_epoch=total_nums, **metric_kwargs)
            score = result['score']
            self.log_info['val_score'] = score
            self.logger.info(f"val log: epoch: {total_nums}, score: {score}")
            self.model.train()

            ckpt = {
                'total_nums': total_nums,
                'wandb_id': self.wandb_run.id,
                'date': datetime.now().isoformat()
            }

            self.save(f'{self.work_dir}/{total_nums}.pth', save_type=WEIGHT, only_model=False, **ckpt)
            os_lib.FileCacher(f'{self.work_dir}/', max_size=max_save_weight_num, stdout_method=self.logger.info).delete_over_range(suffix='pth')

            self.wandb.log(self.log_info)

    def metric(self, *args, real_xs=None, cls_model=None, **kwargs):
        fake_xs = self.predict(*args, **kwargs)
        if real_xs is not None:
            score = image_generation.fid(real_xs, fake_xs, cls_model=cls_model, device=self.device)
            result = dict(score=score)
            return result
        else:
            return {}

    def on_val_step_end(self, vis_image, cur_epoch, visualize, batch_size, max_vis_num, cur_vis_num):
        if visualize:
            n = min(batch_size, max_vis_num - cur_vis_num)
            if n > 0:
                rets = []
                for name, images in vis_image.items():
                    rets.append([{'image': image, '_id': f'{name}.{cur_vis_num}.jpg'} for image in images])

                rets = [r for r in zip(*rets)]
                cache_dir = f'{self.save_result_dir}/{cur_epoch}'
                cache_image = DataVisualizer(cache_dir, verbose=False, pbar=False, stdout_method=self.logger.info)(*rets[:n], return_image=True)
                self.log_info.setdefault('val_image', []).extend([self.wandb.Image(img, caption=Path(r['_id']).stem) for img, r in zip(cache_image, rets[0])])
                cur_vis_num += n
        return cur_vis_num


class GanProcess(IgProcess):
    total_nums = 0

    def model_info(self, **kwargs):
        modules = dict(
            d=self.model.net_d,
            g=self.model.net_g,
        )

        for key, module in modules.items():
            self.logger.info(f'net {key} module info:')
            self._model_info(module, **kwargs)


class Mnist(Process):
    # use `Process(data_dir='data/mnist')` to use digital mnist dataset
    data_dir = 'data/fashion'

    def get_train_data(self):
        from data_parse.cv_data_parse.Mnist import Loader

        loader = Loader(self.data_dir)
        return loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False)[0]

    aug = Apply([
        channel.Gray2BGR(),
        scale.Proportion(),
        pixel_perturbation.MinMax(),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))

        return ret


class WGAN(GanProcess):
    def __init__(self,
                 input_size=64,
                 in_ch=3,
                 hidden_ch=100,
                 model_version='WGAN',
                 **kwargs
                 ):
        from models.image_generate.wgan import Model

        model = Model(
            input_size=input_size,
            in_ch=in_ch,
            hidden_ch=hidden_ch,
        )

        optimizer_d = optim.Adam(model.net_d.parameters(), lr=0.00005, betas=(0.5, 0.999))
        optimizer_g = optim.Adam(model.net_g.parameters(), lr=0.00005, betas=(0.5, 0.999))

        super().__init__(
            model=model,
            optimizer=GanOptimizer(optimizer_d, optimizer_g),
            model_version=model_version,
            input_size=input_size,
            **kwargs
        )

    def fit(self, max_epoch, batch_size, save_period=None, max_save_weight_num=None, metric_kwargs=dict(), **dataloader_kwargs):
        train_dataloader, _, metric_kwargs = self.on_train_start(batch_size, metric_kwargs, return_val_dataloader=True, **dataloader_kwargs)

        self.model.to(self.device)
        optimizer_d, optimizer_g = self.optimizer.optimizer_d, self.optimizer.optimizer_g

        val_noise = torch.normal(mean=0., std=1., size=(64, self.model.hidden_ch, 1, 1), device=self.device)
        if save_period:
            # consider iter_gap
            save_period = int(np.ceil(save_period / 3000)) * 3000

        for i in range(max_epoch):
            self.model.train()
            pbar = tqdm(train_dataloader, desc=f'train {i}/{max_epoch}')
            total_nums = 0
            total_loss_g = 0
            total_loss_d = 0

            for rets in pbar:
                images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images = torch.stack(images)

                loss_d = self.model.loss_d(images)
                loss_d.backward()
                optimizer_d.step()

                total_loss_d += loss_d.item()
                total_nums += len(rets)
                self.total_nums += len(rets)

                # note that, to avoid G so strong, training G once while training D iter_gap times
                if self.total_nums < 1000 or self.total_nums % 20000 < batch_size:
                    iter_gap = 3000
                else:
                    iter_gap = 150

                if self.total_nums % iter_gap < batch_size:
                    loss_g = self.model.loss_g(images)
                    loss_g.backward()
                    optimizer_g.step()

                    total_loss_g += loss_g.item()

                    mean_loss_g = total_loss_g / total_nums
                    mean_loss_d = total_loss_d / total_nums

                    losses = {
                        'mean_loss_d': mean_loss_d,
                        'mean_loss_g': mean_loss_g,
                    }
                    # mem_info = {
                    #     'cpu_info': log_utils.MemoryInfo.get_process_mem_info(),
                    #     'gpu_info': log_utils.MemoryInfo.get_gpu_mem_info()
                    # }

                    pbar.set_postfix({
                        'total_nums': self.total_nums,
                        **losses,
                        # **mem_info
                    })

                    if self.on_train_epoch_end(self.total_nums, save_period, val_noise, batch_size,
                                               losses=losses, max_save_weight_num=max_save_weight_num,
                                               **metric_kwargs):
                        break

    def predict(self, val_noise=None, batch_size=16,
                cur_epoch=-1, model=None, visualize=False, max_vis_num=None, vis_batch_size=64,
                save_ret_func=None, **dataloader_kwargs):
        self.model.to(self.device)
        self.model.eval()
        val_noise = val_noise if val_noise is not None else torch.normal(mean=0., std=1., size=(64, self.model.hidden_ch, 1, 1), device=self.device)
        max_vis_num = max_vis_num or float('inf')
        num_batch = val_noise.shape[0]

        with torch.no_grad():
            fake_xs = []
            for i in tqdm(range(0, num_batch, batch_size)):
                noise_x = val_noise[i: i + batch_size]
                fake_x = self.model.net_g(noise_x)
                fake_xs.append(fake_x)

            fake_xs = torch.cat(fake_xs, 0)
            fake_xs = fake_xs.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

            cur_vis_num = 0
            for i in range(0, num_batch, vis_batch_size):
                fake_x = fake_xs[i: i + vis_batch_size]
                vis_image = dict(
                    fake=fake_x,
                )

                cur_vis_num = self.on_val_step_end(vis_image, cur_epoch, visualize, vis_batch_size, max_vis_num, cur_vis_num)

        return fake_xs


class WGAN_Mnist(WGAN, Mnist):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import WGAN_Mnist as Process

            Process().run(max_epoch=1000, train_batch_size=64, save_period=10000, save_maxsize=10, num_workers=16, metric_kwargs=dict(visualize=True))
    """

    def __init__(self, dataset_version='fashion', **kwargs):
        super().__init__(dataset_version=dataset_version, **kwargs)


class DataProcess(Process):
    aug = Apply([
        scale.Proportion(choice_type=3),
        crop.Random(is_pad=False),
        # scale.LetterBox(),    # there are gray lines
        pixel_perturbation.MinMax(),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret) -> dict:
        # aug = RandomApply([
        #     pixel_perturbation.CutOut([0.25] * 4),
        #     geometry.HFlip(),
        # ], probs=[0.2, 0.5])
        # ret.update(aug(**ret))

        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))

        return ret


class Lsun(DataProcess):
    data_dir = 'data/lsun'
    train_data_num = 20000

    def get_train_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.lsun import Loader

        loader = Loader(self.data_dir)
        iter_data = loader.load(
            set_type=DataRegister.MIX, image_type=DataRegister.ARRAY, generator=False,
            task='cat',
            max_size=self.train_data_num
        )[0]

        # iter_data = loader.load(
        #     set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False,
        #     task='church_outdoor',
        #     max_size=self.train_data_num
        # )[0]

        return iter_data


class CelebA(DataProcess):
    data_dir = 'data/CelebA'
    train_data_num = 40000

    def get_train_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.CelebA import ZipLoader as Loader

        loader = Loader(self.data_dir)
        iter_data = loader.load(
            generator=False,
            img_task='align',
            max_size=self.train_data_num
        )[0]
        return iter_data


class CelebAHQ(DataProcess):
    data_dir = 'data/CelebAHQ'
    train_data_num = 40000

    def get_train_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.CelebAHQ import ZipLoader as Loader

        loader = Loader(self.data_dir)
        iter_data = loader.load(
            generator=False,
            max_size=self.train_data_num
        )[0]
        return iter_data


class StyleGan(GanProcess):
    def __init__(self,
                 model_version='StyleGAN',
                 input_size=128,
                 in_ch=3,
                 **kwargs
                 ):
        from models.image_generate.StyleGAN import Model

        model = Model(
            img_ch=in_ch,
            image_size=input_size,
        )

        generator_params = list(model.net_g.parameters()) + list(model.net_s.parameters())
        optimizer_g = optim.Adam(generator_params, lr=1e-4, betas=(0.5, 0.9))
        optimizer_d = optim.Adam(model.net_d.parameters(), lr=1e-4 * 2, betas=(0.5, 0.9))

        super().__init__(
            model=model,
            optimizer=GanOptimizer(optimizer_d, optimizer_g),
            model_version=model_version,
            input_size=input_size,
            **kwargs
        )

    def model_info(self, **kwargs):
        modules = dict(
            s=self.model.net_s,
            d=self.model.net_d,
            g=self.model.net_g,
        )

        for key, module in modules.items():
            self.logger.info(f'net {key} module info:')
            self._model_info(module, **kwargs)

    def fit(self, max_epoch, batch_size, save_period=None, max_save_weight_num=None,
            vis_num=64 * 8, num_truncate_z=2000,
            per_gp_step=4, per_pp_step=32, min_pp_step=5000,
            metric_kwargs=dict(), **dataloader_kwargs):
        train_dataloader, _, metric_kwargs = self.on_train_start(batch_size, metric_kwargs, return_val_dataloader=False, **dataloader_kwargs)

        optimizer_d, optimizer_g = self.optimizer.optimizer_d, self.optimizer.optimizer_g

        val_obj = (
            self.model.gen_noise_image(vis_num, self.device),
            self.model.gen_same_noise_z_list(vis_num, self.device),
            self.model.gen_noise_z(num_truncate_z, self.device)
        )
        real_xs = []
        cls_model = image_generation.get_default_cls_model(device=self.device)
        steps = 0
        for i in range(max_epoch):
            self.model.train()
            pbar = tqdm(train_dataloader, desc=f'train {i}/{max_epoch}')
            total_nums = 0
            total_loss_g = 0
            total_loss_d = 0

            for rets in pbar:
                images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images = torch.stack(images)

                # train discriminator
                optimizer_d.zero_grad()
                loss_d = self.model.loss_d(images, use_gp=steps % per_gp_step == 0)
                loss_d.backward()
                optimizer_d.step()

                # train generator
                optimizer_g.zero_grad()
                loss_g = self.model.loss_g(images, use_pp=(steps > min_pp_step and steps % per_gp_step == 0))
                loss_g.backward()
                optimizer_g.step()

                steps += 1
                self.total_nums += len(rets)
                total_nums += len(rets)
                total_loss_g += loss_g.item()
                total_loss_d += loss_d.item()
                mean_loss_g = total_loss_g / total_nums
                mean_loss_d = total_loss_d / total_nums

                losses = {
                    'loss_g': loss_g.item(),
                    'loss_d': loss_d.item(),
                    'mean_loss_d': mean_loss_d,
                    'mean_loss_g': mean_loss_g,
                }
                # mem_info = {
                #     'cpu_info': log_utils.MemoryInfo.get_process_mem_info(),
                #     'gpu_info': log_utils.MemoryInfo.get_gpu_mem_info()
                # }

                pbar.set_postfix({
                    'total_nums': self.total_nums,
                    **losses,
                    # **mem_info
                })

                if len(real_xs) < vis_num:
                    images = images.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
                    real_xs.extend(list(images)[:vis_num - len(real_xs)])

                if self.on_train_epoch_end(self.total_nums, save_period, val_obj, batch_size,
                                           losses=losses,
                                           max_save_weight_num=max_save_weight_num,
                                           real_xs=real_xs,
                                           cls_model=cls_model,
                                           **metric_kwargs):
                    break

    @torch.no_grad()
    def predict(self, val_obj=(None, None, None), trunc_psi=0.6, batch_size=16,
                cur_epoch=-1, model=None, visualize=False, max_vis_num=None, vis_batch_size=64,
                save_ret_func=None, **dataloader_kwargs):
        self.model.to(self.device)
        self.model.eval()

        noise_xs, noise_zs, truncate_zs = val_obj
        noise_xs = noise_xs if noise_xs is not None else self.model.gen_noise_image(vis_batch_size, self.device)
        noise_zs = noise_zs if noise_zs is not None else self.model.gen_same_noise_z_list(vis_batch_size, self.device)
        truncate_zs = truncate_zs if truncate_zs is not None else self.model.gen_noise_z(2000, self.device)
        num_batch = noise_xs.shape[0]

        w_styles = []
        for z, num_layer in noise_zs:
            # truncate_style
            truncate_w_style = [self.model.net_s(truncate_zs[i: i + batch_size]) for i in range(0, len(truncate_zs), batch_size)]
            truncate_w_style = torch.cat(truncate_w_style, dim=0).mean(0).unsqueeze(0)
            w_style = self.model.net_s(z)
            w_style = trunc_psi * (w_style - truncate_w_style) + truncate_w_style
            w_styles.append((w_style, num_layer))
        w_styles = torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in w_styles], dim=1)

        max_vis_num = max_vis_num or float('inf')
        cur_vis_num = 0
        fake_xs = []
        for i in range(0, num_batch, batch_size):
            noise_x = noise_xs[i: i + batch_size]
            w_style = w_styles[i: i + batch_size]
            fake_x = self.model.net_g(w_style, noise_x)
            fake_xs.append(fake_x)

        fake_xs = torch.cat(fake_xs, 0)
        fake_xs = fake_xs.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

        for i in range(0, num_batch, vis_batch_size):
            fake_x = fake_xs[i: i + vis_batch_size]
            vis_image = dict(
                fake=fake_x,
            )

            cur_vis_num = self.on_val_step_end(vis_image, cur_epoch, visualize, vis_batch_size, max_vis_num, cur_vis_num)
        return fake_xs


class StyleGan_Mnist(StyleGan, Mnist):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import StyleGan_Mnist as Process

            Process().run(max_epoch=2000, train_batch_size=64, save_period=20000, max_save_weight_num=10, num_workers=16, metric_kwargs=dict(visualize=True))
    """

    def __init__(self, dataset_version='Mnist', input_size=32, **kwargs):
        super().__init__(dataset_version=dataset_version, input_size=input_size, **kwargs)


class StyleGan_Lsun(StyleGan, Lsun):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import StyleGan_Lsun as Process

            Process().run(max_epoch=200, train_batch_size=32, save_period=20000, max_save_weight_num=10, num_workers=16, metric_kwargs=dict(visualize=True))
    """

    def __init__(self, dataset_version='lsun', **kwargs):
        super().__init__(dataset_version=dataset_version, **kwargs)


class StyleGan_CelebA(StyleGan, CelebA):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import StyleGan_CelebA as Process

            Process().run(max_epoch=200, train_batch_size=32, save_period=20000, max_save_weight_num=10, num_workers=16, metric_kwargs=dict(visualize=True))
            {'score': 134.8424}
    """

    def __init__(self, dataset_version='CelebA', **kwargs):
        super().__init__(dataset_version=dataset_version, **kwargs)


class DiProcess(IgProcess):
    total_nums = 0

    def fit(self, max_epoch=100, batch_size=16, save_period=None, max_save_weight_num=None,
            vis_num=64 * 8, num_truncate_z=2000,
            per_gp_step=4, per_pp_step=32, min_pp_step=5000,
            metric_kwargs=dict(), **dataloader_kwargs):
        train_dataloader, _, metric_kwargs = self.on_train_start(batch_size, metric_kwargs, return_val_dataloader=False, **dataloader_kwargs)
        self.model.train()

        val_noise = self.model.gen_x_t(vis_num, device=self.device)

        real_xs = []
        cls_model = image_generation.get_default_cls_model(device=self.device)
        steps = 0
        for i in range(max_epoch):
            pbar = tqdm(train_dataloader, desc=f'train {i}/{max_epoch}')
            total_nums = 0
            total_loss = 0

            for rets in pbar:
                images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images = torch.stack(images)

                self.optimizer.zero_grad()

                output = self.model(images)
                loss = output['loss']
                loss.backward()
                self.optimizer.step()

                steps += 1
                self.total_nums += len(rets)

                total_loss += loss.item()
                total_nums += len(rets)
                mean_loss = total_loss / total_nums

                losses = {
                    'loss': loss.item(),
                    'mean_loss': mean_loss
                }
                # mem_info = {
                #     'cpu_info': log_utils.MemoryInfo.get_process_mem_info(),
                #     'gpu_info': log_utils.MemoryInfo.get_gpu_mem_info()
                # }

                pbar.set_postfix({
                    'total_nums': self.total_nums,
                    **losses,
                    # **mem_info
                })

                if len(real_xs) < vis_num:
                    images = images.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
                    real_xs.extend(list(images)[:vis_num - len(real_xs)])

                if self.on_train_epoch_end(self.total_nums, save_period, val_noise, batch_size,
                                           losses=losses,
                                           max_save_weight_num=max_save_weight_num,
                                           real_xs=real_xs,
                                           cls_model=cls_model,
                                           **metric_kwargs):
                    break

    def predict(self, val_noise=None, batch_size=16,
                cur_epoch=-1, model=None, visualize=False, max_vis_num=None, vis_batch_size=64,
                save_ret_func=None, **dataloader_kwargs):
        self.model.to(self.device)
        self.model.eval()
        val_noise = val_noise if val_noise is not None else self.model.gen_x_t(vis_batch_size, device=self.device)
        max_vis_num = max_vis_num or float('inf')
        num_batch = val_noise.shape[0]

        with torch.no_grad():
            fake_xs = []
            for i in tqdm(range(0, num_batch, batch_size)):
                noise_x = val_noise[i: i + batch_size]
                fake_x = self.model(noise_x)
                fake_xs.append(fake_x)

            fake_xs = torch.cat(fake_xs, 0)
            fake_xs = fake_xs.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

            cur_vis_num = 0
            for i in range(0, num_batch, vis_batch_size):
                fake_x = fake_xs[i: i + vis_batch_size]
                vis_image = dict(
                    fake=fake_x,
                )

                cur_vis_num = self.on_val_step_end(vis_image, cur_epoch, visualize, vis_batch_size, max_vis_num, cur_vis_num)

        return fake_xs


class Ddpm(DiProcess):
    def __init__(self,
                 model_version='Ddpm',
                 input_size=128,
                 in_ch=3,
                 **kwargs
                 ):
        from models.image_generate.ddpm import Model

        model = Model(
            img_ch=in_ch,
            image_size=input_size,
        )
        super().__init__(
            model=model,
            optimizer=optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99)),
            model_version=model_version,
            input_size=input_size,
            **kwargs
        )


class Ddpm_CelebA(Ddpm, CelebA):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import Ddpm_CelebA as Process

            Process().run(max_epoch=200, train_batch_size=32, save_period=20000, max_save_weight_num=10, num_workers=16, metric_kwargs=dict(visualize=True))
            {'score': 134.8424}
    """

    def __init__(self, dataset_version='CelebA', **kwargs):
        super().__init__(dataset_version=dataset_version, **kwargs)


class Dpim(DiProcess):
    def __init__(self,
                 model_version='Ddpm',   # model and train step is same to ddpm, only pred step is different
                 input_size=128,
                 in_ch=3,
                 **kwargs
                 ):
        from models.image_generate.ddim import Model

        model = Model(
            img_ch=in_ch,
            image_size=input_size,
        )
        super().__init__(
            model=model,
            optimizer=optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.99)),
            model_version=model_version,
            input_size=input_size,
            **kwargs
        )


class Ddim_CelebA(Dpim, CelebA):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import Ddpm_CelebA as Process

            Process().run(max_epoch=200, train_batch_size=32, save_period=20000, max_save_weight_num=10, num_workers=16, metric_kwargs=dict(visualize=True))
            {'score': 134.8424}
    """

    def __init__(self, dataset_version='CelebA', **kwargs):
        super().__init__(dataset_version=dataset_version, **kwargs)
