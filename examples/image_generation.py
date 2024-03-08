import cv2
import numpy as np
import torch
from torch import optim, nn
from pathlib import Path
from datetime import datetime
from collections import namedtuple
from utils import os_lib, torch_utils
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, channel, RandomApply, Apply, complex, pixel_perturbation
from data_parse import DataRegister
from data_parse.cv_data_parse.base import DataVisualizer
from processor import Process, DataHooks, bundled, model_process, BatchIterImgDataset, CheckpointHooks


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
    use_early_stop = False
    check_strategy = model_process.STEP
    val_data_num = 64 * 8

    def on_train_start(self, **kwargs):
        from metrics import image_generation

        super().on_train_start(**kwargs)
        self.train_container['metric_kwargs'].update(
            real_x=[],
            fid_cls_model=image_generation.get_default_cls_model(device=self.device)
        )

    def metric(self, real_x=None, fid_cls_model=None, **kwargs):
        from metrics import image_generation

        container = self.predict(**kwargs)
        metric_results = {}
        for name, results in container['model_results'].items():
            if real_x is not None and 'fake_x' in results:
                score = image_generation.fid(real_x, results['fake_x'], cls_model=fid_cls_model, device=self.device)
                result = dict(score=score)
                metric_results[name] = result
            else:
                metric_results[name] = {'score': None}

        return metric_results

    def on_val_reprocess(self, rets, model_results, **kwargs):
        for name, results in model_results.items():
            r = self.val_container['model_results'].setdefault(name, dict())
            for n, items in results.items():
                r.setdefault(n, []).extend(items)

    def on_val_step_end(self, rets, outputs, **kwargs):
        """visualize will work on on_val_end() instead of here,
        because to combine small images into a large image"""

    def on_val_end(self, save_samples=False, save_synth=True, num_synth_per_image=64, is_visualize=False, max_vis_num=None, **kwargs):
        # {name1: {name2: items}}
        for name, results in self.val_container['model_results'].items():
            for name2, items in results.items():
                results[name2] = np.stack(items)

        if is_visualize:
            for i in range(0, self.val_data_num, num_synth_per_image):
                max_vis_num = max_vis_num or float('inf')
                n = min(num_synth_per_image, max_vis_num - self.counters['vis_num'])
                if n > 0:
                    self.visualize(None, self.val_container['model_results'], n,
                                   save_samples=save_samples, save_synth=save_synth,
                                   sub_dir=self.counters["total_nums"], **kwargs)
                    self.counters['vis_num'] += n

    def visualize(self, rets, model_results, n, save_samples=False, save_synth=True,
                  sub_dir='', sub_name='', verbose=False, **kwargs):
        vis_num = self.counters['vis_num']
        for name, results in model_results.items():
            cache_dir = f'{self.cache_dir}/{sub_dir}/{name}'

            if save_synth:
                vis_rets = []
                for name2, images in results.items():
                    vis_rets.append([{'image': image, '_id': f'{sub_name}{name2}.{vis_num}.jpg'} for image in images[vis_num:vis_num + n]])

                vis_rets = [r for r in zip(*vis_rets)]
                cache_image = DataVisualizer(cache_dir, verbose=verbose, pbar=False, stdout_method=self.log)(*vis_rets, return_image=True)
                self.get_log_trace(bundled.WANDB).setdefault(f'val_image/{name}', []).extend(
                    [self.wandb.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=Path(r['_id']).stem) for img, r in zip(cache_image, vis_rets[0])]
                )

            if save_samples:
                cache_dir = f'{cache_dir}/samples'
                os_lib.mk_dir(cache_dir)
                saver = os_lib.Saver(verbose=verbose, stdout_method=self.log)
                for name2, images in results.items():
                    for i in range(vis_num, vis_num + n):
                        image = images[i]
                        saver.save_img(image, f'{cache_dir}/{sub_name}{name2}.{i}.jpg')

                        self.get_log_trace(bundled.WANDB).setdefault(f'val_image/{name}/samples', []).append(
                            self.wandb.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f'{sub_name}{name2}.{i}')
                        )

    def on_predict_step_end(self, model_results, **kwargs):
        for name, results in model_results.items():
            r = self.predict_container['model_results'].setdefault(name, dict())
            for n, items in results.items():
                r.setdefault(n, []).extend(items)

    def on_predict_end(self, is_visualize=False, save_samples=True, save_synth=True, num_synth_per_image=64, save_to_one_dir=True, **kwargs):
        results = [image for image in self.predict_container['model_results'][self.model_name]['fake_x']]
        if is_visualize:
            max_vis_num = len(results)
            date = str(datetime.now().isoformat(timespec='seconds', sep=' '))
            if save_to_one_dir:
                sub_dir = ''
                sub_name = date + '.'
            else:
                sub_dir = date
                sub_name = ''

            for i in range(0, self.val_data_num, num_synth_per_image):
                n = min(num_synth_per_image, max_vis_num - self.counters['vis_num'])
                if n > 0:
                    self.visualize(None, self.predict_container['model_results'], n,
                                   save_samples=save_samples, save_synth=save_synth,
                                   sub_dir=sub_dir, sub_name=sub_name, verbose=True, **kwargs)
                    self.counters['vis_num'] += n

        return results


class GanProcess(IgProcess):
    def model_info(self, **kwargs):
        from utils.torch_utils import ModuleInfo

        modules = dict(
            d=self.model.net_d,
            g=self.model.net_g,
        )

        for key, module in modules.items():
            s, infos = ModuleInfo.std_profile(module, **kwargs)
            self.log(f'net {key} module info:')
            self.log(s)

    def on_backward(self, output, **kwargs):
        """loss backward has been completed in `on_train_step()` already"""
        if hasattr(self, 'ema'):
            self.ema.step(self.model, self.aux_model['ema'])


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

    def data_augment(self, ret, train=True):
        if not train:
            return ret

        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))

        return ret


class WGAN(GanProcess):
    model_version = 'WGAN'
    hidden_ch = 100

    def set_model(self):
        from models.image_generation.wgan import Model
        self.model = Model(
            input_size=self.input_size,
            in_ch=self.in_ch,
            hidden_ch=self.hidden_ch,
        )

    def set_optimizer(self):
        optimizer_d = optim.Adam(self.model.net_d.parameters(), lr=0.00005, betas=(0.5, 0.999))
        optimizer_g = optim.Adam(self.model.net_g.parameters(), lr=0.00005, betas=(0.5, 0.999))
        self.optimizer = GanOptimizer(optimizer_d, optimizer_g)

    def on_train_step(self, rets, batch_size=16, **kwargs) -> dict:
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

        real_x = self.train_container['metric_kwargs']['real_x']
        if len(real_x) < self.val_data_num:
            images = images.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            real_x.extend(list(images)[:self.val_data_num - len(real_x)])

        return {
            'loss.g': loss_g,
            'loss.d': loss_d,
        }

    def on_train_step_end(self, *args, check_period=None, **kwargs):
        if check_period:
            # consider iter_gap
            check_period = int(np.ceil(check_period / 3000)) * 3000

        return super().on_train_step_end(*args, check_period=check_period, **kwargs)

    def get_val_data(self, *args, **kwargs):
        val_obj = self.model.gen_noise(self.val_data_num, self.device)
        return val_obj

    def on_val_start(self, val_dataloader=None, batch_size=16, dataloader_kwargs=dict(), **kwargs):
        val_noise = val_dataloader if val_dataloader is not None else self.get_val_data()
        num_batch = val_noise.shape[0]

        def gen():
            for i in range(0, num_batch, batch_size):
                yield val_noise[i: i + batch_size]

        super().on_val_start(val_dataloader=gen(), **kwargs)

    def on_val_step(self, rets, **kwargs) -> dict:
        noise_x = rets
        model_results = {}
        for name, model in self.models.items():
            fake_x = model.net_g(noise_x)
            fake_x = fake_x.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

            model_results[name] = dict(
                fake_x=fake_x,
            )

        return model_results


class WGAN_Mnist(WGAN, Mnist):
    """
    Usage:
        .. code-block:: python

            from examples.image_generation import WGAN_Mnist as Process

            Process().run(max_epoch=1000, train_batch_size=64, fit_kwargs=dict(check_period=40000, max_save_weight_num=10), metric_kwargs=dict(is_visualize=True))
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

    def data_augment(self, ret, train=True) -> dict:
        if not train:
            return ret

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
    train_data_num = 50000

    input_size = 128
    in_ch = 3

    def get_train_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.lsun import Loader

        loader = Loader(self.data_dir)

        return loader.load(
            set_type=DataRegister.MIX, image_type=DataRegister.ARRAY, generator=False,
            task='cat',
            max_size=self.train_data_num
        )[0]


class CelebA(DataProcess):
    dataset_version = 'CelebA'
    data_dir = 'data/CelebA'
    train_data_num = 40000  # do not set too large, 'cause images will be cached in memory
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


class IterCelebA(DataProcess):
    train_dataset_ins = BatchIterImgDataset

    dataset_version = 'CelebA'
    data_dir = 'data/CelebA'
    train_data_num = None
    input_size = 128
    in_ch = 3

    def get_train_data(self, *args, **kwargs):
        """before get data, run the following script first

        from data_parse.cv_data_parse.CelebA import ZipLoader as Loader, DataRegister
        from data_parse.cv_data_parse.SimpleGridImage import Saver

        loader = Loader(data_dir)
        saver = Saver(data_dir)

        per_images = 100
        batch_rets = []
        for i, ret in enumerate(loader.load(generator=True,img_task='align')[0]):
            batch_rets.append(ret)
            if i % per_images == per_images - 1:
                saver([[batch_rets]], task='img_align_celeba_bind')
                batch_rets = []
        """
        from data_parse.cv_data_parse.SimpleGridImage import Loader

        loader = Loader(self.data_dir, image_suffix='jpg')
        return lambda: loader.load(generator=False, task='img_align_celeba_bind', size=(178, 218), max_size=self.train_data_num)[0]


class CelebAHQ(DataProcess):
    dataset_version = 'CelebAHQ'
    data_dir = 'data/CelebAHQ'
    train_data_num = 40000  # do not set too large, 'cause images will be cached in memory

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
        from models.image_generation.StyleGAN import Model

        self.model = Model(
            img_ch=self.in_ch,
            image_size=self.input_size,
        )
        self.warmup()

    def warmup(self):
        """init some params"""
        self.model.net_d(torch.randn((1, self.in_ch, self.input_size, self.input_size)))

    def set_optimizer(self):
        generator_params = list(self.model.net_g.parameters()) + list(self.model.net_s.parameters())
        optimizer_g = optim.Adam(generator_params, lr=1e-4, betas=(0.5, 0.9))
        optimizer_d = optim.Adam(self.model.net_d.parameters(), lr=1e-4 * 2, betas=(0.5, 0.9))

        self.optimizer = GanOptimizer(optimizer_d, optimizer_g)

    def model_info(self, **kwargs):
        from utils.torch_utils import ModuleInfo

        modules = dict(
            s=self.model.net_s,
            d=self.model.net_d,
            g=self.model.net_g,
        )

        for key, module in modules.items():
            s, infos = ModuleInfo.std_profile(module, **kwargs)
            self.log(f'net {key} module info:')
            self.log(s)

    per_gp_step = 4
    per_pp_step = 32
    min_pp_step = 5000

    def on_train_step(self, rets, **kwargs) -> dict:
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
            use_pp=(self.counters['total_steps'] > self.min_pp_step and self.counters['total_steps'] % self.per_pp_step == 0)
        )
        loss_g.backward()
        self.optimizer.optimizer_g.step()

        real_x = self.train_container['metric_kwargs']['real_x']
        if len(real_x) < self.val_data_num:
            images = images.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            real_x.extend(list(images)[:self.val_data_num - len(real_x)])

        return {
            'loss.g': loss_g,
            'loss.d': loss_d,
        }

    num_truncate_z = 2000

    def get_val_data(self, *args, **kwargs):
        val_obj = (
            self.model.gen_noise_image(self.val_data_num, self.device),
            self.model.gen_same_noise_z_list(self.val_data_num, self.device),
            self.model.gen_noise_z(self.num_truncate_z, self.device)
        )

        return val_obj

    def on_val_start(self, val_dataloader=(None, None, None), batch_size=16, trunc_psi=0.6, **kwargs):
        model = self.model

        noise_xs, noise_zs, truncate_zs = val_dataloader if val_dataloader is not None else self.get_val_data()
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

        def gen():
            for i in range(0, num_batch, batch_size):
                noise_x = noise_xs[i: i + batch_size]
                w_style = w_styles[i: i + batch_size]
                yield noise_x, w_style

        super().on_val_start(val_dataloader=gen(), **kwargs)

    def on_val_step(self, rets, vis_batch_size=64, **kwargs) -> dict:
        noise_x, w_style = rets
        model_results = {}
        for name, model in self.models.items():
            fake_x = model.net_g(w_style, noise_x)
            fake_x = fake_x.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

            model_results[name] = dict(
                fake_x=fake_x,
            )

        return model_results


class StyleGan_Mnist(StyleGan, Mnist):
    """
    Usage:
        .. code-block:: python

            from examples.image_generation import StyleGan_Mnist as Process

            Process().run(
                max_epoch=200, train_batch_size=32,
                fit_kwargs=dict(check_period=40000, max_save_weight_num=10),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
    """


class StyleGan_Lsun(StyleGan, Lsun):
    """
    Usage:
        .. code-block:: python

            from examples.image_generation import StyleGan_Lsun as Process

            Process().run(
                max_epoch=100, train_batch_size=32,
                fit_kwargs=dict(check_period=40000, max_save_weight_num=10),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
    """


class StyleGan_CelebA(StyleGan, CelebA):
    """
    Usage:
        .. code-block:: python

            from examples.image_generation import StyleGan_CelebA as Process

            Process().run(
                max_epoch=50, train_batch_size=32,
                fit_kwargs=dict(check_period=40000, max_save_weight_num=10),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
            {'score': 134.8424}
    """


class StyleGan_IterCelebA(StyleGan, IterCelebA):
    """
    Usage:
        .. code-block:: python

            from examples.image_generation import StyleGan_IterCelebA as Process

            Process().run(
                max_epoch=10, train_batch_size=32,
                fit_kwargs=dict(check_period=40000, max_save_weight_num=10, dataloader_kwargs=dict(shuffle=False, drop_last=True, num_workers=16)),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
            {'score': 63.01491}
    """


class VAE(IgProcess):
    model_version = 'VAE'

    def set_model(self):
        from models.image_generation.VAE import Model, Config
        self.model = Model(
            img_ch=self.in_ch,
            image_size=self.input_size,
        )

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.99))

    def on_train_step(self, rets, **kwargs):
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        images = torch.stack(images)
        output = self.model(images)

        real_x = self.train_container['metric_kwargs']['real_x']
        if len(real_x) < self.val_data_num:
            images = images.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            real_x.extend(list(images)[:self.val_data_num - len(real_x)])

        return output

    def get_val_data(self, *args, **kwargs):
        """use real_x"""

    def on_val_start(self, val_dataloader=None, batch_size=16, dataloader_kwargs=dict(), **kwargs):
        def gen():
            for i in range(0, self.val_data_num, batch_size):
                yield self.train_container['metric_kwargs']['real_x'][i: i + batch_size]

        super().on_val_start(val_dataloader=gen(), **kwargs)

    def on_val_step(self, rets, **kwargs) -> dict:
        images = [torch.from_numpy(ret).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        real_x = torch.stack(images).permute(0, 3, 1, 2)
        model_results = {}
        for name, model in self.models.items():
            fake_x = model(real_x)
            fake_x = fake_x.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            model_results[name] = dict(
                fake_x=fake_x,
            )

        return model_results


class VAE_CelebA(VAE, CelebA):
    """
    Usage:
        .. code-block:: python

            from examples.image_generation import Ddpm_CelebA as Process

            Process().run(
                max_epoch=50, train_batch_size=32,
                fit_kwargs=dict(check_period=40000, max_save_weight_num=10),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
    """


class DiProcess(IgProcess):
    def on_train_step(self, rets, **kwargs) -> dict:
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
        images = torch.stack(images)
        images = images * 2 - 1  # normalize, [0, 1] -> [-1, 1]
        output = self.model(images)

        real_x = self.train_container['metric_kwargs']['real_x']
        if len(real_x) < self.val_data_num:
            images = images.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            real_x.extend(list(images)[:self.val_data_num - len(real_x)])

        return output

    def get_val_data(self, *args, **kwargs):
        val_obj = self.model.gen_x_t(self.val_data_num)
        return val_obj

    def on_val_start(self, val_dataloader=None, batch_size=16, dataloader_kwargs=dict(), **kwargs):
        val_noise = val_dataloader if val_dataloader is not None else self.get_val_data()

        def gen():
            for i in range(0, self.val_data_num, batch_size):
                yield val_noise[i: i + batch_size]

        super().on_val_start(val_dataloader=gen(), **kwargs)

    def on_val_step(self, rets, **kwargs) -> dict:
        noise_x = rets
        model_results = {}
        for name, model in self.models.items():
            fake_x = model(noise_x)
            fake_x = (fake_x + 1) * 0.5  # unnormalize, [-1, 1] -> [0, 1]
            fake_x = fake_x.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            model_results[name] = dict(
                fake_x=fake_x,
            )

        return model_results


class Ddpm(DiProcess):
    model_version = 'Ddpm'

    def set_model(self):
        from models.image_generation.ddpm import Model
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

            from examples.image_generation import Ddpm_CelebA as Process

            Process().run(
                max_epoch=50, train_batch_size=32,
                fit_kwargs=dict(check_period=40000, max_save_weight_num=10),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
    """


class Dpim(DiProcess):
    # model and train step is same to ddpm, only pred step is different
    # so still use ddpm to name the model
    model_version = 'Ddpm'

    def set_model(self):
        from models.image_generation.ddim import Model
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

            from examples.image_generation import Ddpm_CelebA as Process

            Process().run(
                max_epoch=50, train_batch_size=32,
                fit_kwargs=dict(check_period=40000, max_save_weight_num=10),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
            {'score': 64.1675}
    """


class Ddim_CelebAHQ(Dpim, CelebAHQ):
    """
    Usage:
        .. code-block:: python

            from examples.image_generation import Ddim_CelebAHQ as Process

            Process().run(
                max_epoch=50, train_batch_size=32,
                fit_kwargs=dict(check_period=40000, max_save_weight_num=10),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
    """


class Ldm(DiProcess):
    model_version = 'ldm'
    in_ch = 3
    input_size = 512

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.99))

    aug = Apply([
        scale.Rectangle(),
        channel.BGR2RGB(),
        channel.HWC2CHW(),
    ])

    def val_data_augment(self, ret) -> dict:
        if 'image' in ret and ret['image'] is not None:
            ret.update(dst=self.input_size)
            ret.update(self.aug(**ret))
        return ret

    def on_val_step(self, rets, **kwargs) -> dict:
        texts = []
        neg_texts = []
        images = []
        for ret in rets:
            if 'text' in ret:
                texts.append(ret['text'])
            if 'image' in ret and ret['image'] is not None:
                images.append(torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float))
            if 'neg_text' in ret and ret['neg_text']:
                neg_texts.append(ret['neg_text'])

        neg_texts = neg_texts if neg_texts else None

        if images:
            images = torch.stack(images)
            images /= 255.
            images = images * 2 - 1  # normalize, [0, 1] -> [-1, 1]
        else:
            images = None

        model_results = {}
        for name, model in self.models.items():
            # note, something wrong with autocast, got inf result
            # with torch.cuda.amp.autocast(True):
            fake_x = model(x=images, text=texts, neg_text=neg_texts)
            fake_x = (fake_x + 1) * 0.5  # unnormalize, [-1, 1] -> [0, 1]
            fake_x = fake_x.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            model_results[name] = dict(
                fake_x=fake_x,
            )

        return model_results

    model_input_template = namedtuple('model_inputs', ['text', 'neg_text', 'image'], defaults=[None, None])

    def gen_predict_inputs(self, *objs, images=None, **kwargs):
        assert len(objs) <= 2

        if len(objs) == 2:
            pos_texts, neg_texts = objs
            assert len(pos_texts) == len(neg_texts)
        else:
            pos_texts = objs[0]
            neg_texts = [None] * len(pos_texts)

        if images:
            if not isinstance(images, (list, tuple)):
                images = [images for _ in pos_texts]
        else:
            images = [None] * len(pos_texts)

        rets = []
        for text, neg_text, image in zip(pos_texts, neg_texts, images):
            if isinstance(image, str):
                image = os_lib.Loader(verbose=False).load_img(image)
            rets.append(self.model_input_template(image=image, text=text, neg_text=neg_text)._asdict())

        return rets

    def on_predict_step_end(self, model_results, add_watermark=True, watermark='watermark', **kwargs):
        for name, results in model_results.items():
            r = self.predict_container['model_results'].setdefault(name, dict())
            for n, items in results.items():
                items[..., :] = items[..., ::-1]  # note, official model output is Image type, must convert to cv2 type
                if add_watermark:
                    self.add_watermark(items, watermark)
                r.setdefault(n, []).extend(items)

    def add_watermark(self, images, watermark='watermark'):
        """be safe, add watermark for images
        see https://github.com/ShieldMnt/invisible-watermark"""
        from imwatermark import WatermarkEncoder

        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', watermark.encode('utf-8'))

        images = [wm_encoder.encode(image, 'dwtDct') for image in images]
        return images


class LoadSDPretrain(CheckpointHooks):
    def load_pretrain(self):
        if hasattr(self, 'pretrain_model'):
            from models.image_generation.ldm import convert_weights

            state_dict = torch_utils.Load.from_file(self.pretrain_model)
            if torch_utils.WeightsFormats.get_format_from_suffix(self.pretrain_model) == 'PyTorch':
                state_dict = state_dict['state_dict']
            state_dict = convert_weights(state_dict)
            self.model.load_state_dict(state_dict, strict=False)


class SDv1(Ldm):
    model_version = 'sd'
    model_sub_version = 'vanilla'  # for config choose
    dataset_version = model_sub_version
    cond_pretrain_model = 'openai/clip-vit-large-patch14'

    def set_model(self):
        from models.image_generation.sdv1 import Model, Config
        self.model = Model(
            img_ch=self.in_ch,
            image_size=self.input_size,
            cond_config=dict(
                pretrain_model=self.cond_pretrain_model,
                # if having ldm pretrain_model, do not download the clip weight file, only the config file
                # 'cause the ldm pretrain_model contains the clip weight
                load_weight=not hasattr(self, 'pretrain_model')
            ),
            **Config.get(self.model_sub_version)
        )


class SDv1Pretrained(SDv1, LoadSDPretrain):
    """no training, only for prediction

    Usage:
        .. code-block:: python

            from examples.image_generation import SDv1Pretrained as Process

            process = Process(pretrain_model='...', cond_pretrain_model='...', model_sub_version='...')
            process.init()

            # txt2img
            prompt = 'a painting of a virus monster playing guitar'
            neg_prompt = ''
            prompts = ['a painting of a virus monster playing guitar', 'a painting of two virus monster playing guitar']
            neg_prompts = ['', '']

            # predict one
            image = process.single_predict(prompt, neg_prompt, is_visualize=True)

            # predict batch
            images = process.batch_predict(prompts, neg_prompts, batch_size=2, is_visualize=True)

            # img2img
            image = 'test.jpg'
            images = ['test1.jpg', 'test2.jpg']

            # predict one
            image = process.single_predict(prompt, image, is_visualize=True)

            # predict batch
            images = process.batch_predict(prompts, neg_prompt, images=image, batch_size=2, is_visualize=True)     # base on same image
            images = process.batch_predict(prompts, neg_prompts, images=images, batch_size=2, is_visualize=True)    # base on different image
    """


class SDv2(Ldm):
    model_version = 'sd'
    model_sub_version = 'v2'  # for config choose
    dataset_version = model_sub_version
    cond_pretrain_model = None
    input_size = 768

    def set_model(self):
        from models.image_generation.sdv2 import Model, Config

        self.model = Model(
            img_ch=self.in_ch,
            image_size=self.input_size,
            cond_config=dict(
                # if having ldm pretrain_model, set `self.cond_pretrain_model=None`,
                # do not download the clip weight file, only the config file
                # 'cause the ldm pretrain_model contains the clip weight
                pretrain_model=self.cond_pretrain_model,
            ),
            **Config.get(self.model_sub_version)
        )


class SDv2Pretrained(SDv2, LoadSDPretrain):
    """no training, only for prediction

    Usage:
        .. code-block:: python

            from examples.image_generation import SDv2Pretrained as Process

            # same to `SDv1Pretrained`
            ...
    """


class SDXL(Ldm):
    model_version = 'sd'
    model_sub_version = 'xl_base'  # for config choose
    dataset_version = model_sub_version
    cond_pretrain_model = None
    input_size = 1024
    low_memory_run = False

    def set_model(self):
        from models.image_generation.sdxl import Model, Config

        self.model = Model(
            img_ch=self.in_ch,
            image_size=self.input_size,
            **Config.get(self.model_sub_version)
        )

        if self.low_memory_run:
            from functools import partial

            for module in [self.model.cond, self.model.backbone, self.model.vae]:
                module.__call__ = partial(torch_utils.ModuleManager.low_memory_run, module, self.device)


class SDXLPretrained(SDXL, LoadSDPretrain):
    """no training, only for prediction

    Usage:
        .. code-block:: python

            from examples.image_generation import SDXLPretrained as Process

            # same to `SDv1Pretrained`
            ...
    """
