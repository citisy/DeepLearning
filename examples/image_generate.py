import time
import cv2
import numpy as np
import torch
from torch import optim, nn
from metrics import image_generation
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, channel, RandomApply, Apply, complex, pixel_perturbation
from data_parse import DataRegister
from pathlib import Path
from data_parse.cv_data_parse.base import DataVisualizer
from processor import Process, DataHooks, bundled, BaseDataset, model_process
from utils import configs, cv_utils, os_lib, log_utils
from datetime import datetime


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
    vis_num = 64 * 8

    def on_train_start(self, container, **kwargs):
        super().on_train_start(container, **kwargs)
        container['real_xs'] = []
        container['cls_model'] = image_generation.get_default_cls_model(device=self.device)

    def _on_train_step_end(self, container, save_period=None, batch_size=None, max_save_weight_num=None, **kwargs):
        total_nums = self.counters['total_nums']
        if save_period and total_nums % save_period < batch_size:
            self.trace({'total_nums': total_nums}, bundled.WANDB)

            losses = container.get('losses')
            if losses is not None:
                for k, v in losses.items():
                    self.trace({f'loss/{k}': v}, bundled.WANDB)
                    if np.isnan(v) or np.isinf(v):
                        container['end_flag'] = True
                        self.log(f'Train will be stop soon, got {v} value from {k}')

            epoch_start_time = container.get('epoch_start_time')
            if epoch_start_time is not None:
                self.trace({'time_consume': (time.time() - epoch_start_time) / 60}, bundled.WANDB)

            result = self.metric(
                real_xs=container.get('real_xs'),
                cls_model=container.get('cls_model'),
                val_dataloader=container.get('val_dataloader'),
                **container.get('metric_kwargs', {})
            )
            score = result['score']
            self.logger.info(f"val log: epoch: {total_nums}, score: {score}")

            self.set_mode(train=True)

            ckpt = {
                'optimizer': self.optimizer.state_dict(),
                'counters': self.counters,
                'wandb_id': self.wandb_id,
                'date': datetime.now().isoformat()
            }

            self.save(f'{self.work_dir}/last.pth', save_type=model_process.WEIGHT, save_items=ckpt)
            os_lib.FileCacher(f'{self.work_dir}/', max_size=max_save_weight_num, stdout_method=self.log).delete_over_range(suffix='pth')
            self.log_trace(bundled.WANDB)

    def on_train_epoch_end(self, container, **kwargs) -> bool:
        return container.get('end_flag', False)

    def metric(self, real_xs=None, cls_model=None, **kwargs):
        container = self.predict(**kwargs)
        if real_xs is not None and 'fake_xs' in container['preds']:
            score = image_generation.fid(real_xs, container['preds']['fake_xs'], cls_model=cls_model, device=self.device)
            result = dict(score=score)
            return result
        else:
            return {'score': None}

    def on_val_start(self, container, model=None, **kwargs):
        self.set_mode(train=False)
        self.counters['vis_num'] = 0
        self.counters.setdefault('epoch', -1)
        container['preds'] = {}

    def on_val_step_end(self, rets, outputs, container, **kwargs):
        for name, images in outputs.items():
            images = images.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            container['preds'].setdefault(name, []).append(images)

    def on_val_end(self, container, num_synth_per_image=64, is_visualize=False, max_vis_num=None, **kwargs):
        preds = container['preds']

        for k, v in preds.items():
            preds[k] = np.concatenate(v, 0)
            num_batch = preds[k].shape[0]

        for i in range(0, num_batch, num_synth_per_image):
            vis_image = {k: v[i: i + num_synth_per_image] for k, v in preds.items()}
            self._on_val_end(container, is_visualize=is_visualize, vis_image=vis_image, num_synth_per_image=num_synth_per_image, max_vis_num=max_vis_num, **kwargs)

        super().on_val_end(container, **kwargs)

    def _on_val_end(self, container, is_visualize=False, vis_image={}, num_synth_per_image=64, max_vis_num=None, **kwargs):
        if is_visualize:
            max_vis_num = max_vis_num or float('inf')
            n = min(num_synth_per_image, max_vis_num - self.counters['vis_num'])
            if n > 0:
                rets = []
                for name, images in vis_image.items():
                    rets.append([{'image': image, '_id': f'{name}.{self.counters["vis_num"]}.jpg'} for image in images])

                rets = [r for r in zip(*rets)]
                cache_dir = f'{self.cache_dir}/{self.counters["epoch"]}'
                cache_image = DataVisualizer(cache_dir, verbose=False, pbar=False, stdout_method=self.logger.info)(*rets[:n], return_image=True)
                self.get_log_trace(bundled.WANDB).setdefault('val_image', []).extend(
                    [self.wandb.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=Path(r['_id']).stem) for img, r in zip(cache_image, rets[0])]
                )
                self.counters['vis_num'] += n


class GanProcess(IgProcess):
    def model_info(self, **kwargs):
        modules = dict(
            d=self.model.net_d,
            g=self.model.net_g,
        )

        for key, module in modules.items():
            self.logger.info(f'net {key} module info:')
            self._model_info(module, **kwargs)

    def on_train_epoch_start(self, container, **kwargs):
        _counters = ('per_epoch_loss', 'per_epoch_nums', 'epoch', 'total_loss_g', 'total_loss_d')
        super().on_train_epoch_start(container, _counters=_counters, **kwargs)

    def on_backward(self, output, container, **kwargs):
        """has been completed in on_train_step() already"""

    def on_train_step_end(self, rets, output, container, more_log=False, save_period=None, batch_size=None, max_save_weight_num=None, **kwargs):
        self.counters['total_nums'] += len(rets)
        self.counters['total_steps'] += 1
        self.counters['per_epoch_nums'] += len(rets)

        losses = {}
        for k, v in output.items():
            if k.startswith('loss'):
                losses[k] = v.item()

        self.counters['total_loss_g'] += losses['loss.g']
        self.counters['total_loss_d'] += losses['loss.d']
        mean_loss_g = self.counters['total_loss_g'] / self.counters['per_epoch_nums']
        mean_loss_d = self.counters['total_loss_d'] / self.counters['per_epoch_nums']

        losses.update({
            'mean_loss_d': mean_loss_d,
            'mean_loss_g': mean_loss_g,
        })

        mem_info = {
            'cpu_info': log_utils.MemoryInfo.get_process_mem_info(),
            'gpu_info': log_utils.MemoryInfo.get_gpu_mem_info()
        } if more_log else {}

        self.log({
            'total_nums': self.counters['total_nums'],
            **losses,
            **mem_info
        }, 'pbar')

        self._on_train_step_end(container, save_period=save_period, batch_size=batch_size, max_save_weight_num=max_save_weight_num, **kwargs)


class Mnist(DataHooks):
    # use `Process(data_dir='data/mnist')` to use digital mnist dataset
    dataset_version = 'fashion'
    data_dir = 'data/fashion'

    input_size = 64
    in_ch = 3

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

    def train_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))

        return ret


class WGAN(GanProcess):
    model_version = 'WGAN'
    hidden_ch = 100

    def set_model(self):
        from models.image_generate.wgan import Model
        self.model = Model(
            input_size=self.input_size,
            in_ch=self.in_ch,
            hidden_ch=self.hidden_ch,
        )

    def set_optimizer(self):
        optimizer_d = optim.Adam(self.model.net_d.parameters(), lr=0.00005, betas=(0.5, 0.999))
        optimizer_g = optim.Adam(self.model.net_g.parameters(), lr=0.00005, betas=(0.5, 0.999))
        self.optimizer = GanOptimizer(optimizer_d, optimizer_g)

    def on_train_step(self, rets, container, batch_size=16, **kwargs) -> dict:
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        images = torch.stack(images)

        loss_d = self.model.loss_d(images)
        loss_d.backward()
        self.optimizer.optimizer_d.step()
        self.optimizer.optimizer_d.zero_grad()

        # note that, to avoid G so strong, training G once while training D iter_gap times
        if 0 < self.counters['total_nums'] < 1000 or self.counters['total_nums'] % 20000 < batch_size:
            iter_gap = 3000
        else:
            iter_gap = 150

        loss_g = torch.tensor(0, device=self.device)
        if self.counters['total_nums'] % iter_gap < batch_size:
            loss_g = self.model.loss_g(images)
            loss_g.backward()
            self.optimizer.optimizer_g.step()
            self.optimizer.optimizer_g.zero_grad()

        real_xs = container['real_xs']
        if len(real_xs) < self.vis_num:
            images = images.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            real_xs.extend(list(images)[:self.vis_num - len(real_xs)])

        return {
            'loss.g': loss_g,
            'loss.d': loss_d,
        }

    def on_train_step_end(self, *args, save_period=None, **kwargs):
        if save_period:
            # consider iter_gap
            save_period = int(np.ceil(save_period / 3000)) * 3000

        super().on_train_step_end(*args, save_period=save_period, **kwargs)

    def get_val_data(self, *args, **kwargs):
        val_obj = self.model.gen_noise(self.vis_num, self.device)
        return val_obj

    def on_val_start(self, container, val_dataloader=None, batch_size=16, dataloader_kwargs=dict(), **kwargs):
        super().on_val_start(container, **kwargs)

        val_noise = val_dataloader if val_dataloader is not None else torch.normal(mean=0., std=1., size=(batch_size, self.model.hidden_ch, 1, 1), device=self.device)
        num_batch = val_noise.shape[0]

        def gen():
            for i in range(0, num_batch, batch_size):
                yield val_noise[i: i + batch_size]

        container['val_dataloader'] = gen()

    def on_val_step(self, rets, container, **kwargs) -> tuple:
        noise_x = rets
        fake_x = self.model.net_g(noise_x)

        fake_x = fake_x.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        outputs = dict(
            fake=fake_x,
        )
        container['fake_xs'].append(fake_x)

        return rets, outputs


class WGAN_Mnist(WGAN, Mnist):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import WGAN_Mnist as Process

            Process().run(max_epoch=1000, train_batch_size=64, save_period=10000, save_maxsize=10, metric_kwargs=dict(is_visualize=True))
    """


class DataProcess(DataHooks):
    rand_aug = RandomApply([
        pixel_perturbation.CutOut([0.25] * 4),
        geometry.HFlip(),
    ], probs=[0.2, 0.5])

    aug = Apply([
        scale.Proportion(choice_type=3),
        crop.Random(is_pad=False),
        # scale.LetterBox(),    # there are gray lines
    ])

    post_aug = Apply([
        pixel_perturbation.MinMax(),
        # pixel_perturbation.Normalize(127.5, 127.5),
        channel.HWC2CHW()
    ])

    def train_data_augment(self, ret) -> dict:
        # ret.update(self.rand_aug(**ret))
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))
        ret.update(self.post_aug(**ret))

        return ret

    def val_data_restore(self, ret) -> dict:
        ret = self.post_aug.restore(ret)
        ret.update(pixel_perturbation.Clip()(**ret))
        return ret

    def get_val_dataloader(self, **dataloader_kwargs):
        return self.get_val_data()


class Lsun(DataProcess):
    dataset_version = 'lsun'
    data_dir = 'data/lsun'
    train_data_num = 20000

    input_size = 128
    in_ch = 3

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
    dataset_version = 'CelebA'
    data_dir = 'data/CelebA'
    train_data_num = 40000
    input_size = 128
    in_ch = 3

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
    dataset_version = 'CelebAHQ'
    data_dir = 'data/CelebAHQ'
    train_data_num = 40000

    input_size = 1024
    in_ch = 3

    def get_train_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.CelebAHQ import ZipLoader as Loader

        loader = Loader(self.data_dir)
        iter_data = loader.load(
            generator=False,
            max_size=self.train_data_num
        )[0]
        return iter_data


class StyleGan(GanProcess):
    model_version = 'StyleGAN'

    def set_model(self):
        from models.image_generate.StyleGAN import Model

        self.model = Model(
            img_ch=self.in_ch,
            image_size=self.input_size,
        )

    def set_optimizer(self):
        generator_params = list(self.model.net_g.parameters()) + list(self.model.net_s.parameters())
        optimizer_g = optim.Adam(generator_params, lr=1e-4, betas=(0.5, 0.9))
        optimizer_d = optim.Adam(self.model.net_d.parameters(), lr=1e-4 * 2, betas=(0.5, 0.9))

        self.optimizer = GanOptimizer(optimizer_d, optimizer_g)

    def model_info(self, **kwargs):
        modules = dict(
            s=self.model.net_s,
            d=self.model.net_d,
            g=self.model.net_g,
        )

        for key, module in modules.items():
            self.logger.info(f'net {key} module info:')
            self._model_info(module, **kwargs)

    per_gp_step = 4
    per_pp_step = 32
    min_pp_step = 5000

    def on_train_step(self, rets, container, vis_num=64 * 8, **kwargs) -> dict:
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        images = torch.stack(images)

        # train discriminator
        self.optimizer.optimizer_d.zero_grad()
        loss_d = self.model.loss_d(
            images,
            use_gp=self.counters['total_steps'] % self.per_gp_step == 0
        )
        loss_d.backward()
        self.optimizer.optimizer_d.step()

        # train generator
        self.optimizer.optimizer_g.zero_grad()
        loss_g = self.model.loss_g(
            images,
            use_pp=(self.counters['total_steps'] > self.min_pp_step and self.counters['total_steps'] % self.per_gp_step == 0)
        )
        loss_g.backward()
        self.optimizer.optimizer_g.step()

        real_xs = container['real_xs']
        if len(real_xs) < vis_num:
            images = images.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            real_xs.extend(list(images)[:vis_num - len(real_xs)])

        return {
            'loss.g': loss_g,
            'loss.d': loss_d,
        }

    num_truncate_z = 2000

    def get_val_data(self, *args, **kwargs):
        val_obj = (
            self.model.gen_noise_image(self.vis_num, self.device),
            self.model.gen_same_noise_z_list(self.vis_num, self.device),
            self.model.gen_noise_z(self.num_truncate_z, self.device)
        )

        return val_obj

    def on_val_start(self, container, val_dataloader=(None, None, None), batch_size=16, trunc_psi=0.6, dataloader_kwargs=dict(), **kwargs):
        super().on_val_start(container, **kwargs)
        model = self.model

        noise_xs, noise_zs, truncate_zs = val_dataloader if val_dataloader is not None else self.get_val_dataloader(batch_size=batch_size, **dataloader_kwargs)

        noise_xs = noise_xs if noise_xs is not None else model.gen_noise_image(batch_size, self.device)
        noise_zs = noise_zs if noise_zs is not None else model.gen_same_noise_z_list(batch_size, self.device)
        truncate_zs = truncate_zs if truncate_zs is not None else model.gen_noise_z(2000, self.device)
        num_batch = noise_xs.shape[0]

        w_styles = []
        for z, num_layer in noise_zs:
            # truncate_style
            truncate_w_style = [model.net_s(truncate_zs[i: i + batch_size]) for i in range(0, len(truncate_zs), batch_size)]
            truncate_w_style = torch.cat(truncate_w_style, dim=0).mean(0).unsqueeze(0)
            w_style = model.net_s(z)
            w_style = trunc_psi * (w_style - truncate_w_style) + truncate_w_style
            w_styles.append((w_style, num_layer))
        w_styles = torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in w_styles], dim=1)
        container['w_styles'] = w_styles
        container['noise_xs'] = noise_xs

        def gen():
            for i in range(0, num_batch, batch_size):
                noise_x = noise_xs[i: i + batch_size]
                w_style = w_styles[i: i + batch_size]
                yield noise_x, w_style

        container['val_dataloader'] = gen()

    def on_val_step(self, rets, container, vis_batch_size=64, **kwargs) -> tuple:
        noise_x, w_style = rets
        fake_x = self.model.net_g(w_style, noise_x)
        fake_x = fake_x.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        outputs = dict(
            fake=fake_x,
        )
        container['fake_xs'].append(fake_x)

        return rets, outputs


class StyleGan_Mnist(StyleGan, Mnist):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import StyleGan_Mnist as Process

            Process().run(max_epoch=2000, train_batch_size=64, save_period=20000, max_save_weight_num=10, metric_kwargs=dict(is_visualize=True))
    """


class StyleGan_Lsun(StyleGan, Lsun):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import StyleGan_Lsun as Process

            Process().run(max_epoch=200, train_batch_size=32, save_period=20000, max_save_weight_num=10, metric_kwargs=dict(is_visualize=True))
    """


class StyleGan_CelebA(StyleGan, CelebA):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import StyleGan_CelebA as Process

            Process().run(max_epoch=200, train_batch_size=32, save_period=20000, max_save_weight_num=10, metric_kwargs=dict(is_visualize=True))
            {'score': 134.8424}
    """


class DiProcess(IgProcess):
    def on_train_step(self, rets, container, **kwargs) -> dict:
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        images = torch.stack(images)
        output = self.model(images)

        real_xs = container['real_xs']
        if len(real_xs) < self.vis_num:
            images = images.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            real_xs.extend(list(images)[:self.vis_num - len(real_xs)])

        return output

    def get_val_data(self, *args, **kwargs):
        val_obj = self.model.gen_x_t(self.vis_num, device=self.device)
        return val_obj

    def on_val_start(self, container, val_dataloader=None, batch_size=16, dataloader_kwargs=dict(), **kwargs):
        super().on_val_start(container, **kwargs)
        val_noise = val_dataloader if val_dataloader is not None else self.model.gen_x_t(batch_size, device=self.device)
        num_batch = val_noise.shape[0]

        def gen():
            for i in range(0, num_batch, batch_size):
                yield val_noise[i: i + batch_size]

        container['val_dataloader'] = gen()

    def on_val_step(self, rets, container, **kwargs) -> tuple:
        noise_x = rets
        fake_x = self.model(noise_x)
        fake_x = fake_x.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
        outputs = dict(
            fake=fake_x,
        )
        container['fake_xs'].append(fake_x)

        return rets, outputs


class Ddpm(DiProcess):
    model_version = 'Ddpm'

    def set_model(self):
        from models.image_generate.ddpm import Model
        self.model = Model(
            img_ch=self.in_ch,
            image_size=self.input_size,
        )

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.99))


class Ddpm_CelebA(Ddpm, CelebA):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import Ddpm_CelebA as Process

            Process().run(max_epoch=200, train_batch_size=32, save_period=20000, max_save_weight_num=10, metric_kwargs=dict(is_visualize=True))
    """


class Dpim(DiProcess):
    # model and train step is same to ddpm, only pred step is different
    # so still use ddpm to name the model
    model_version = 'Ddpm'

    def set_model(self):
        from models.image_generate.ddim import Model
        self.model = Model(
            img_ch=self.in_ch,
            image_size=self.input_size,
        )

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.99))


class Ddim_CelebA(Dpim, CelebA):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import Ddpm_CelebA as Process

            Process().run(max_epoch=200, train_batch_size=32, save_period=20000, max_save_weight_num=10, metric_kwargs=dict(is_visualize=True))
            {'score': 64.1675}
    """


class Ddim_CelebAHQ(Dpim, CelebAHQ):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import Ddim_CelebAHQ as Process

            Process().run(max_epoch=200, train_batch_size=4, save_period=20000, max_save_weight_num=10, metric_kwargs=dict(is_visualize=True))
    """
