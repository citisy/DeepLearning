from collections import namedtuple
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Iterable

import cv2
import numpy as np
import torch
from torch import optim
from torch.utils.data import Dataset
from torch import nn

from data_parse import DataRegister
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, channel, RandomApply, Apply, pixel_perturbation, Lambda
from data_parse.cv_data_parse.datasets.base import DataVisualizer
from models import normalizations
from processor import Process, DataHooks, bundled, model_process, BatchIterImgDataset, CheckpointHooks
from utils import os_lib, torch_utils, configs


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
    val_data_num = 64 * 8

    input_size: int
    in_ch: int

    use_fid_cls_model = True
    fid_cls_model: 'nn.Module'

    def on_train_start(self, **kwargs):
        loop_objs, process_kwargs = super().on_train_start(**kwargs)
        loop_objs['metric_kwargs'].update(
            real_x=[],
        )

        def set_fid_cls_model():
            if self.use_fid_cls_model and not hasattr(self, 'fid_cls_model'):
                from metrics import image_generation

                self.fid_cls_model = image_generation.get_default_cls_model(device=self.device)

        self.register_val_start(set_fid_cls_model)
        return loop_objs, process_kwargs

    def metric(self, real_x=None, **kwargs):
        from metrics import image_generation

        container = self.predict(**kwargs)
        metric_results = {}
        for name, results in container['model_results'].items():
            if real_x is not None and 'fake_x' in results:
                if self.use_fid_cls_model:
                    score = image_generation.fid(real_x, results['fake_x'], cls_model=self.fid_cls_model, device=self.device)
                else:
                    score = -1

                result = dict(score=score)
                metric_results[name] = result
            else:
                metric_results[name] = {'score': -1}

        return metric_results

    def on_val_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        for model_name, results in model_results.items():
            r = process_results.setdefault(model_name, dict())
            for data_name, items in results.items():
                r.setdefault(data_name, []).extend(items)

    def on_val_step_end(self, loop_objs, is_visualize=False, save_samples=True, save_to_one_dir=True, **kwargs):
        model_results = loop_objs['model_results']

        if is_visualize and save_samples:
            sub_dir, sub_name = self._make_save_obj(save_to_one_dir)
            for model_name, results in model_results.items():
                for data_name, items in results.items():
                    for i, image in enumerate(items):
                        cache_dir = f'{self.cache_dir}/{sub_dir}/{model_name}'
                        image_save_stem = f'{sub_name}{data_name}.{i}'
                        self.visualize_one(image, cache_dir, model_name, image_save_stem, **kwargs)

    def on_val_end(self, loop_objs, process_results=dict(), save_synth=True, num_synth_per_image=64, is_visualize=False, max_vis_num=None, **kwargs):
        if is_visualize and save_synth:
            results = [image for image in process_results[self.model_name]['fake_x']]
            max_vis_num = len(results)
            vis_num = 0

            for i in range(0, self.val_data_num, num_synth_per_image):
                n = min(num_synth_per_image, max_vis_num - vis_num)
                if n > 0:
                    self.visualize_synth(process_results, n, vis_num=vis_num, **kwargs)
                vis_num += n

    def _make_save_obj(self, save_to_one_dir):
        date = str(datetime.now().isoformat(timespec='seconds', sep=' '))
        if save_to_one_dir:
            sub_dir = ''
            sub_name = date + '.'
        else:
            sub_dir = date
            sub_name = ''

        return sub_dir, sub_name

    def on_predict_reprocess(
            self, loop_objs, process_results=dict(),
            return_outputs=True, add_watermark=False, **kwargs
    ):
        if add_watermark:
            model_results = loop_objs['model_results']
            for model_name, results in model_results.items():
                for data_name, items in results.items():
                    results[data_name] = self.add_watermark(items, **kwargs)

        if return_outputs:
            self.on_val_reprocess(loop_objs, process_results=process_results, **kwargs)

    def add_watermark(self, images, watermark='watermark', **kwargs):
        """be safe, add watermark for images
        see https://github.com/ShieldMnt/invisible-watermark"""
        from imwatermark import WatermarkEncoder

        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark('bytes', watermark.encode('utf-8'))

        images = [wm_encoder.encode(image, 'dwtDct') for image in images]
        return images

    def visualize_one(self, image, cache_dir, model_name='', image_save_stem='', verbose=False, **kwargs):
        cache_dir = f'{cache_dir}/samples'
        os_lib.mk_dir(cache_dir)
        saver = os_lib.Saver(verbose=verbose, stdout_method=self.log)
        saver.save_img(image, f'{cache_dir}/{image_save_stem}.jpg')

        self.get_log_trace(bundled.WANDB).setdefault(f'val_image/{model_name}/samples', []).append(
            self.wandb.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f'{image_save_stem}')
        )

    def on_predict_end(
            self,
            return_outputs=True, process_results=dict(),
            **kwargs
    ):
        if return_outputs:
            results = [image for image in process_results[self.model_name]['fake_x']]
            self.on_val_end(None, process_results=process_results, **kwargs)
            return results

    def visualize_synth(self, model_results, n, vis_num=0, save_to_one_dir=True, verbose=False, **kwargs):
        sub_dir, sub_name = self._make_save_obj(save_to_one_dir)
        for model_name, results in model_results.items():
            cache_dir = f'{self.cache_dir}/{sub_dir}/{model_name}'
            vis = DataVisualizer(cache_dir, verbose=verbose, pbar=False, stdout_method=self.log)
            vis_rets = []
            for data_name, images in results.items():
                vis_rets.append([{'image': image, '_id': f'{sub_name}{data_name}.{vis_num}.jpg'} for image in images[vis_num:vis_num + n]])

            vis_rets = [r for r in zip(*vis_rets)]
            cache_image = vis(*vis_rets, return_image=True)
            self.get_log_trace(bundled.WANDB).setdefault(f'val_image/{model_name}', []).extend(
                [self.wandb.Image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=Path(r['_id']).stem) for img, r in zip(cache_image, vis_rets[0])]
            )


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

    def on_backward(self, loop_objs, use_ema=False, **kwargs):
        """loss backward has been completed in `on_train_step()` already"""
        if use_ema:
            self.ema.step()


class Mnist(DataHooks):
    # use `Process(data_dir='data/mnist')` to use digital mnist dataset
    dataset_version = 'fashion'
    data_dir = 'data/fashion'

    input_size = 64
    in_ch = 3

    def get_train_data(self):
        from data_parse.cv_data_parse.datasets.Mnist import Loader

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

    def set_optimizer(self, lr_g=0.00005, betas_g=(0.5, 0.999), lr_d=0.00005, betas_d=(0.5, 0.999), **kwargs):
        optimizer_d = optim.Adam(self.model.net_d.parameters(), lr=lr_d, betas=betas_d)
        optimizer_g = optim.Adam(self.model.net_g.parameters(), lr=lr_g, betas=betas_g)
        self.optimizer = GanOptimizer(optimizer_d, optimizer_g)

    def on_train_step(self, loop_objs, batch_size=16, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
        images = torch.stack(images)

        loss_d = self.model.loss_d(images)
        loss_d.backward()
        self.optimizer.optimizer_d.step()
        self.optimizer.optimizer_d.zero_grad()

        # note that, to avoid G so strong, training G once while training D iter_gap times
        if 0 < loop_objs['total_nums'] < 1000 or loop_objs['total_nums'] % 20000 < batch_size:
            iter_gap = 3000
        else:
            iter_gap = 150

        loss_g = torch.tensor(0, device=self.device)
        if loop_objs['total_nums'] % iter_gap < batch_size:
            loss_g = self.model.loss_g(images)
            loss_g.backward()
            self.optimizer.optimizer_g.step()
            self.optimizer.optimizer_g.zero_grad()

        real_x = loop_objs['metric_kwargs']['real_x']
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

        return super().on_val_start(val_dataloader=gen(), **kwargs)

    def on_val_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        noise_x = loop_inputs
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

            from bundles.image_generation import WGAN_Mnist as Process

            Process().run(
                max_epoch=1000,
                train_batch_size=64,
                fit_kwargs=dict(
                    check_period=40000,
                    check_strategy='step',
                    max_save_weight_num=10,
                ),
                metric_kwargs=dict(is_visualize=True)
            )
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
        from data_parse.cv_data_parse.datasets.lsun import Loader

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
        from data_parse.cv_data_parse.datasets.CelebA import ZipLoader as Loader

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
        from data_parse.cv_data_parse.datasets.SimpleGridImage import Loader

        loader = Loader(self.data_dir, image_suffix='.jpg')
        return lambda: loader.load(generator=False, task='img_align_celeba_bind', size=(178, 218), max_size=self.train_data_num)[0]


class CelebAHQ(DataProcess):
    dataset_version = 'CelebAHQ'
    data_dir = 'data/CelebAHQ'
    train_data_num = 40000  # do not set too large, 'cause images will be cached in memory

    input_size = 1024
    in_ch = 3

    def get_train_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.datasets.CelebAHQ import ZipLoader as Loader

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

    def set_optimizer(self, lr_g=1e-4, betas_g=(0.5, 0.9), lr_d=1e-4 * 2, betas_d=(0.5, 0.9), **kwargs):
        generator_params = list(self.model.net_g.parameters()) + list(self.model.net_s.parameters())
        optimizer_g = optim.Adam(generator_params, lr=lr_g, betas=betas_g)
        optimizer_d = optim.Adam(self.model.net_d.parameters(), lr=lr_d, betas=betas_d)

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

    def on_train_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
        images = torch.stack(images)

        # train discriminator
        self.optimizer.optimizer_d.zero_grad()
        loss_d = self.model.loss_d(
            images,
            use_gp=loop_objs['total_steps'] % self.per_gp_step == 0
        )
        loss_d.backward()
        self.optimizer.optimizer_d.step()

        # train generator
        self.optimizer.optimizer_g.zero_grad()
        loss_g = self.model.loss_g(
            images,
            use_pp=(loop_objs['total_steps'] > self.min_pp_step and loop_objs['total_steps'] % self.per_pp_step == 0)
        )
        loss_g.backward()
        self.optimizer.optimizer_g.step()

        real_x = loop_objs['metric_kwargs']['real_x']
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

        return super().on_val_start(val_dataloader=gen(), **kwargs)

    def on_val_step(self, loop_objs, vis_batch_size=64, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        noise_x, w_style = loop_inputs
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

            from bundles.image_generation import StyleGan_Mnist as Process

            Process().run(
                max_epoch=200, train_batch_size=32,
                fit_kwargs=dict(
                    check_period=40000,
                    check_strategy='step',
                    max_save_weight_num=10,
                ),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
    """


class StyleGan_Lsun(StyleGan, Lsun):
    """
    Usage:
        .. code-block:: python

            from bundles.image_generation import StyleGan_Lsun as Process

            Process().run(
                max_epoch=100, train_batch_size=32,
                fit_kwargs=dict(
                    check_period=40000,
                    check_strategy='step',
                    max_save_weight_num=10,
                ),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
    """


class StyleGan_CelebA(StyleGan, CelebA):
    """
    Usage:
        .. code-block:: python

            from bundles.image_generation import StyleGan_CelebA as Process

            Process().run(
                max_epoch=50, train_batch_size=32,
                fit_kwargs=dict(
                    check_period=40000,
                    check_strategy='step',
                    max_save_weight_num=10,
                ),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
            {'score': 134.8424}
    """


class StyleGan_IterCelebA(StyleGan, IterCelebA):
    """
    Usage:
        .. code-block:: python

            from bundles.image_generation import StyleGan_IterCelebA as Process

            Process().run(
                max_epoch=10, train_batch_size=32,
                fit_kwargs=dict(
                    check_period=40000,
                    check_strategy='step',
                    max_save_weight_num=10,
                    dataloader_kwargs=dict(shuffle=False, drop_last=True, num_workers=16)
                ),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
            {'score': 63.01491}
    """


class VAE(IgProcess):
    model_version = 'VAE'

    def set_optimizer(self, lr=1e-4, betas=(0.9, 0.99), **kwargs):
        super().set_optimizer(lr=lr, betas=betas, **kwargs)

    def set_model(self):
        from models.image_generation.VAE import Model
        self.model = Model(
            img_ch=self.in_ch,
            image_size=self.input_size,
        )

    def on_train_step(self, loop_objs, **kwargs):
        loop_inputs = loop_objs['loop_inputs']
        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
        images = torch.stack(images)
        output = self.model(images)

        real_x = loop_objs['metric_kwargs']['real_x']
        if len(real_x) < self.val_data_num:
            images = images.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            real_x.extend(list(images)[:self.val_data_num - len(real_x)])

        return output

    def get_val_data(self, *args, **kwargs):
        """use real_x"""

    def on_val_start(self, val_dataloader=None, batch_size=16, dataloader_kwargs=dict(), real_x=[], **kwargs):
        def gen():
            for i in range(0, self.val_data_num, batch_size):
                yield real_x[i: i + batch_size]

        super().on_val_start(val_dataloader=gen(), **kwargs)

    def on_val_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        images = [torch.from_numpy(ret).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
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

            from bundles.image_generation import Ddpm_CelebA as Process

            Process().run(
                max_epoch=50, train_batch_size=32,
                fit_kwargs=dict(
                    check_period=40000,
                    check_strategy='step',
                    max_save_weight_num=10
                ),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
    """


def _load_images(images, b, start_idx, end_idx):
    if images:
        if not isinstance(images, (list, tuple)):
            # base on one image
            images = [images for _ in range(b)]
        else:
            images = images[start_idx: end_idx]
        images = [os_lib.loader.load_img(image) if isinstance(image, str) else image for image in images]
    else:
        images = [None] * b

    return images


class DiProcess(IgProcess):
    low_memory_run = True
    use_half = True

    def set_model_status(self):
        if self.use_pretrained:
            self.load_pretrained()
        if self.low_memory_run:
            self.model._device = self.device  # explicitly define the device for the model
            self.model.set_low_memory_run()

        else:
            if not isinstance(self.device, list):
                self.model.to(self.device)

        if self.use_half:
            self.model.set_half()
            # note, if model init from meta, official weight of norm is float16, force to float32
            torch_utils.ModuleManager.apply(
                self.model,
                lambda module: module.to(torch.float),
                include=[normalizations.GroupNorm32],
            )
        else:
            self.model.to(torch.float)

    def set_optimizer(self, lr=1e-4, betas=(0.9, 0.99), **kwargs):
        super().set_optimizer(lr=lr, betas=betas, **kwargs)

    def get_model_inputs(self, loop_inputs, train=True):
        if train:
            images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
            images = torch.stack(images)
            images = images * 2 - 1  # normalize, [0, 1] -> [-1, 1]
            model_inputs = dict(
                x=images
            )
        else:
            model_inputs = dict(
                x=loop_inputs
            )
        return model_inputs

    def on_train_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=True)
        with torch.cuda.amp.autocast(True):
            output = self.model(**model_inputs)

        real_x = loop_objs['metric_kwargs']['real_x']
        if len(real_x) < self.val_data_num:
            images = model_inputs['x']
            images = (images + 1) * 0.5
            images = images.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            real_x.extend(list(images)[:self.val_data_num - len(real_x)])

        return output

    def get_val_data(self, *args, **kwargs):
        val_obj = self.model.gen_x_t(self.val_data_num)
        return val_obj

    def on_val_start(self, val_dataloader=None, batch_size=16, dataloader_kwargs=dict(), **kwargs):
        iter_data = val_dataloader if val_dataloader is not None else self.get_val_data()

        def gen():
            for i in range(0, min(self.val_data_num, len(iter_data)), batch_size):
                yield iter_data[i: i + batch_size]

        super().on_val_start(val_dataloader=gen(), batch_size=batch_size, **kwargs)

    def on_val_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)
        model_inputs.update(model_kwargs)

        model_results = {}
        for name, model in self.models.items():
            # note, something wrong with autocast, got inf result
            # with torch.cuda.amp.autocast(True):
            fake_x = model(**model_inputs)
            fake_x = (fake_x + 1) * 0.5  # unnormalize, [-1, 1] -> [0, 1]
            fake_x = fake_x.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            model_results[name] = dict(
                fake_x=fake_x,
            )

        return model_results

    def on_predict_reprocess(self, loop_objs, **kwargs):
        model_results = loop_objs['model_results']
        for model_name, results in model_results.items():
            for data_name, items in results.items():
                items[..., :] = items[..., ::-1]  # note, official model output is Image type, must convert to cv2 type
                results[data_name] = items

        return super().on_predict_reprocess(loop_objs, **kwargs)


class Ddpm(DiProcess):
    model_version = 'Ddpm'

    def set_model(self):
        from models.image_generation.ddpm import Model
        self.model = Model(
            img_ch=self.in_ch,
            image_size=self.input_size,
        )


class Ddpm_CelebA(Ddpm, CelebA):
    """
    Usage:
        .. code-block:: python

            from bundles.image_generation import Ddpm_CelebA as Process

            Process().run(
                max_epoch=50, train_batch_size=32,
                fit_kwargs=dict(
                    check_period=40000,
                    check_strategy='step',
                    max_save_weight_num=10
                ),
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


class Ddim_CelebA(Dpim, CelebA):
    """
    Usage:
        .. code-block:: python

            from bundles.image_generation import Ddpm_CelebA as Process

            Process().run(
                max_epoch=50, train_batch_size=32,
                fit_kwargs=dict(
                    check_period=40000,
                    check_strategy='step',
                    max_save_weight_num=10
                ),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
            {'score': 64.1675}
    """


class Ddim_CelebAHQ(Dpim, CelebAHQ):
    """
    Usage:
        .. code-block:: python

            from bundles.image_generation import Ddim_CelebAHQ as Process

            Process().run(
                max_epoch=50, train_batch_size=32,
                fit_kwargs=dict(
                    check_period=40000,
                    check_strategy='step',
                    max_save_weight_num=10
                ),
                metric_kwargs=dict(is_visualize=True, max_vis_num=64 * 8),
            )
    """


class WithLora(Process):
    use_lora = False
    use_half_lora = True
    lora_wrap: 'models.tuning.lora.ModelWrap'
    lora_pretrained_model: str
    lora_config = {}

    def init_components(self):
        super().init_components()
        if self.use_lora:
            self.set_lora()
            self.load_lora_pretrain()

    def unset_lora(self):
        self.lora_wrap.dewrap()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['lora'] = self.lora_wrap.state_dict()
        return state_dict

    def save_lora_weight(self, suffix, max_save_weight_num, **kwargs):
        fp = f'{self.work_dir}/{suffix}.lora.safetensors'
        torch_utils.Export.to_safetensors(self.lora_wrap.state_dict(), fp)
        os_lib.FileCacher(self.work_dir, max_size=max_save_weight_num, stdout_method=self.log).delete_over_range(suffix=r'\d+\.lora\.safetensors')
        self.log(f'Successfully save lora to {fp}!')


class WithSDLora(WithLora):
    config_version = 'v1'  # for config choose

    def set_lora(self):
        from models.tuning import lora

        self.lora_wrap = lora.ModelWrap(
            include=(
                'attn_res.fn.to_qkv',
                'ff_res.fn',
                'transformer_blocks',
                'proj_in',
                'proj_out',
                'to_out.linear'
            ),
            exclude=(
                'drop',
                'act',
                'norm',
                'view',
                'ff.1',
                'attend'
            ),
            **self.lora_config
        )
        self.lora_wrap.wrap(self.model)

        for full_name in self.lora_wrap.layers:
            layer = torch_utils.ModuleManager.get_module_by_name(self.model, full_name)
            layer.to(self.device)
            if self.use_half_lora:
                layer.to(torch.bfloat16)

        self.register_save_checkpoint(self.save_lora_weight)
        self.log('Successfully add lora!')

    def load_lora_pretrain(self):
        if hasattr(self, 'lora_pretrained_model'):
            if 'v1' in self.config_version:
                from models.image_generation.sdv1 import WeightLoader, WeightConverter
            elif 'v2' in self.config_version:
                from models.image_generation.sdv2 import WeightLoader, WeightConverter
            elif 'xl' in self.config_version:
                from models.image_generation.sdxl import WeightLoader, WeightConverter
            else:
                raise

            state_dict = WeightLoader.auto_load(self.lora_pretrained_model)
            state_dict = WeightConverter.from_official_lora(state_dict)
            self.lora_wrap.load_state_dict(state_dict, strict=True)
            self.log(f'Loaded lora pretrain model from {self.lora_pretrained_model}')


class WithSDControlNet(Process):
    use_control_net = False
    control_net_wrap: 'models.tuning.control_net.ModelWrap'
    control_net_pretrained_model: str
    control_net_config = {}
    control_net_version = 'v1.5'  # for config choose

    def init_components(self):
        super().init_components()
        if self.use_control_net:
            self.set_control_net()
            self.load_control_net_pretrain()

    def set_control_net(self):
        from models.tuning.control_net import ModelWrap, Config

        config = Config.get(self.control_net_version)
        config = configs.ConfigObjParse.merge_dict(config, self.control_net_config)
        self.control_net_wrap = ModelWrap(config)
        self.control_net_wrap.wrap(self.model)
        self.log('Successfully add control_net!')

    def load_control_net_pretrain(self):
        if hasattr(self, 'control_net_pretrained_model'):
            if 'v1' in self.control_net_version:
                from models.image_generation.sdv1 import WeightConverter
            elif 'v2' in self.control_net_version:
                from models.image_generation.sdv2 import WeightConverter
            elif 'xl' in self.control_net_version:
                from models.image_generation.sdxl import WeightConverter
            else:
                raise

            state_dict = torch_utils.Load.from_file(self.control_net_pretrained_model)
            state_dict = WeightConverter.from_official_controlnet(state_dict)
            self.control_net_wrap.load_state_dict(state_dict, strict=False)
            self.log(f'Loaded control_net pretrain model from {self.control_net_pretrained_model}')

    control_aug = Apply([
        Lambda(lambda image, **kwargs: cv2.Canny(image, 100, 200)),
        scale.RuderLetterBox(),
        channel.Keep3Dims(),
        channel.Keep3Channels(),
        channel.HWC2CHW(),
    ])

    def gen_predict_inputs(self, *objs, control_images=None, start_idx=None, end_idx=None, **kwargs):
        rets = super().gen_predict_inputs(*objs, start_idx=start_idx, end_idx=end_idx, **kwargs)
        b = len(rets)
        control_images = _load_images(control_images, b, start_idx, end_idx)
        for i, ret in enumerate(rets):
            ret['control_image'] = control_images[i]

        return rets

    def val_data_augment(self, ret) -> dict:
        ret = super().val_data_augment(ret)
        if 'control_image' in ret and ret['control_image'] is not None:
            ret['control_image'] = self.control_aug(image=ret['control_image'], dst=self.input_size)['image']
        return ret

    def get_model_val_inputs(self, loop_inputs):
        model_inputs = super().get_model_val_inputs(loop_inputs)
        control_images = []
        for ret in loop_inputs:
            if 'control_image' in ret and ret['control_image'] is not None:
                control_image = ret.pop('control_image')
                control_images.append(torch.from_numpy(control_image).to(self.device, non_blocking=True, dtype=torch.float))

        if control_images:
            control_images = torch.stack(control_images)
            control_images /= 255
            model_inputs.update(control_images=control_images)

        return model_inputs


class FromSDPretrained(CheckpointHooks):
    config_version = 'v1'  # for config choose

    def load_pretrained(self):
        if 'v1' in self.config_version:
            from models.image_generation.sdv1 import WeightLoader, WeightConverter
        elif 'v2' in self.config_version:
            from models.image_generation.sdv2 import WeightLoader, WeightConverter
        elif 'xl' in self.config_version:
            from models.image_generation.sdxl import WeightLoader, WeightConverter
        else:
            raise

        state_dict = WeightLoader.auto_load(self.pretrained_model)
        state_dict = WeightConverter.from_official(state_dict)
        self.model.load_state_dict(state_dict, strict=False, assign=True)
        self.log(f'load pretrain model from {self.pretrained_model}')

    @classmethod
    def from_pretrained(cls, pretrained_model, **kwargs):
        raise NotImplemented


class BaseSD(DiProcess):
    model_version = 'sd'
    config_version = 'v1'  # for config choose
    in_ch = 3
    input_size = 512

    train_cond = False

    model_config: dict = {}

    def set_model(self):
        if 'v1' in self.config_version:
            from models.image_generation.sdv1 import Model, Config

            if not hasattr(self, 'encoder_fn'):
                self.encoder_fn = 'openai/clip-vit-large-patch14/merges.txt'
            if not hasattr(self, 'vocab_fn'):
                self.vocab_fn = 'openai/clip-vit-large-patch14/vocab.json'

        elif 'v2' in self.config_version:
            from models.image_generation.sdv2 import Model, Config

            if not hasattr(self, 'encoder_fn'):
                self.encoder_fn = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K/merges.txt'
            if not hasattr(self, 'vocab_fn'):
                self.vocab_fn = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K/vocab.json'

        elif 'xl' in self.config_version:
            from models.image_generation.sdxl import Model, Config

            if not hasattr(self, 'encoder_fn'):
                self.encoder_fn = 'openai/clip-vit-large-patch14/merges.txt'
            if not hasattr(self, 'vocab_fn'):
                self.vocab_fn = 'openai/clip-vit-large-patch14/vocab.json'

        else:
            raise

        if 'v2' in self.config_version:
            # note, special pad_id for laion clip model
            self.tokenizer.pad_id = 0
        elif 'xl' in self.config_version:
            # todo: different pad_id from sdv1 and sdv2
            pass

        model_config = configs.ConfigObjParse.merge_dict(Config.get(self.config_version), self.model_config)
        with torch.device('meta'):  # fast to init model
            self.model = Model(
                img_ch=self.in_ch,
                image_size=self.input_size,
                **model_config
            )

    def set_tokenizer(self):
        from data_parse.nl_data_parse.pre_process.bundled import CLIPTokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(self.vocab_fn, self.encoder_fn)

    def get_model_inputs(self, loop_inputs, train=True):
        if train:
            return self.get_model_train_inputs(loop_inputs)
        else:
            return self.get_model_val_inputs(loop_inputs)


class SDTrainer(BaseSD):
    def set_optimizer(self, **kwargs):
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.99))
        # todo, found that, it will take device 0 when `optimizer.step()`, even thought only choose device 1
        # it's bug for `bnb.optim.AdamW8bit`, found no resolution to fix yet
        self.optimizer = torch_utils.make_optimizer_cls('AdamW8bit')(self.model.parameters(), lr=1e-4)

    def get_model_train_inputs(self, loop_inputs):
        texts = [ret['text'] for ret in loop_inputs]
        inputs = self.tokenizer.encode_attention_paragraphs(texts)
        text_ids = torch.tensor(inputs['segments_ids']).to(self.device)
        text_weights = torch.tensor(inputs['segments_weights']).to(self.device)

        images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
        images = torch.stack(images)
        images = images * 2 - 1  # normalize, [0, 1] -> [-1, 1]
        return dict(
            x=images,
            text_ids=text_ids,
            text_weights=text_weights
        )


class SDPredictor(BaseSD):
    def get_val_data(self, *args, pos_text_fn='prompts.txt', neg_text_fn='neg_prompts.txt', **kwargs) -> Optional[Iterable | Dataset | List[Dataset]]:
        pos_texts = os_lib.loader.load_txt(f'{self.data_dir}/{pos_text_fn}')
        neg_texts = os_lib.loader.load_txt(f'{self.data_dir}/{neg_text_fn}')

        if self.val_data_num:
            pos_texts = pos_texts[:self.val_data_num]
            neg_texts = neg_texts[:self.val_data_num]

        iter_data = []
        for text, neg_text in zip(pos_texts, neg_texts):
            iter_data.append(dict(text=text, neg_text=neg_text))

        return iter_data

    val_aug = Apply([
        scale.Rectangle(),
        channel.HWC2CHW(),
    ])

    def val_data_augment(self, ret) -> dict:
        if 'image' in ret and ret['image'] is not None:
            ret.update(dst=self.input_size)
            ret.update(self.val_aug(**ret))
        return ret

    def get_model_val_inputs(self, loop_inputs):
        texts = []
        neg_texts = []
        images = []
        mask_images = []
        for ret in loop_inputs:
            if 'text' in ret:
                texts.append(ret['text'])

            if 'neg_text' in ret and ret['neg_text'] is not None:
                neg_texts.append(ret['neg_text'])
            else:
                neg_texts.append('')

            if 'image' in ret and ret['image'] is not None:
                image = ret.pop('image')
                if image.shape[0] == 4:
                    image, mask_image = image[:-1], image[-1:]
                    if 'mask_image' not in ret or ret['mask_image'] is None:
                        mask_images.append(torch.from_numpy(mask_image).to(self.device, non_blocking=True, dtype=torch.float))

                images.append(torch.from_numpy(image).to(self.device, non_blocking=True, dtype=torch.float))

            if 'mask_image' in ret and ret['mask_image'] is not None:
                mask_image = ret.pop('mask_image')
                mask_images.append(torch.from_numpy(mask_image).to(self.device, non_blocking=True, dtype=torch.float))

        inputs = self.tokenizer.encode_attention_paragraphs(texts)
        text_ids = torch.tensor(inputs['segments_ids']).to(self.device)
        text_weights = torch.tensor(inputs['segments_weights']).to(self.device)

        neg_inputs = self.tokenizer.encode_attention_paragraphs(neg_texts)
        neg_text_ids = torch.tensor(neg_inputs['segments_ids']).to(self.device)
        neg_text_weights = torch.tensor(neg_inputs['segments_weights']).to(self.device)

        if images:
            images = torch.stack(images)
            images /= 255.
            images = images * 2 - 1  # normalize, [0, 1] -> [-1, 1]

        if mask_images:
            mask_images = torch.stack(mask_images)
            mask_images /= 255

        return dict(
            text_ids=text_ids,
            neg_text_ids=neg_text_ids,
            text_weights=text_weights,
            neg_text_weights=neg_text_weights,
            x=images,
            mask_x=mask_images,
        )

    model_input_template = namedtuple('model_inputs', ['text', 'neg_text', 'image', 'mask_image'], defaults=[None, None])

    def gen_predict_inputs(self, *objs, neg_texts=None, images=None, mask_images=None, start_idx=None, end_idx=None, **kwargs):
        """

        Args:
            *objs:
            neg_texts (str|List[str]):
            images (str|List[str]|np.ndarray|List[np.ndarray]):
            mask_images (str|List[str]|np.ndarray|List[np.ndarray]):
            start_idx:
            end_idx:
            **kwargs:

        Returns:

        """
        pos_texts = objs[0][start_idx: end_idx]
        b = len(pos_texts)

        if neg_texts is None:
            neg_texts = [None] * b
        elif isinstance(neg_texts, str):
            neg_texts = [neg_texts] * b
        else:
            neg_texts = neg_texts[start_idx: end_idx]

        images = _load_images(images, b, start_idx, end_idx)
        mask_images = _load_images(mask_images, b, start_idx, end_idx)

        rets = []
        for text, neg_text, image, mask_image in zip(pos_texts, neg_texts, images, mask_images):
            rets.append(self.model_input_template(image=image, text=text, neg_text=neg_text, mask_image=mask_image)._asdict())

        return rets


class SD(WithSDLora, WithSDControlNet, FromSDPretrained, SDTrainer, SDPredictor):
    """no training, only for prediction

    Usage:
        .. code-block:: python

            from bundles.image_generation import SD as Process

            process = Process(
                pretrained_model='...',
                vocab_fn='xxx/vocab.json',
                encoder_fn='xxx/merges.txt',
                config_version='...',

                low_memory_run=True,
                use_half=True,

                # if using lora
                # use_lora=True,
                # lora_pretrained_model='xxx',
            )
            process.init()

            # txt2img
            prompt = 'a painting of a virus monster playing guitar'
            neg_prompt = ''
            prompts = ['a painting of a virus monster playing guitar', 'a painting of two virus monster playing guitar']
            neg_prompts = ['', '']

            # predict one
            image = process.single_predict(prompt, neg_texts=neg_prompt, is_visualize=True)

            # predict batch
            images = process.batch_predict(prompts, neg_texts=neg_prompts, batch_size=2, is_visualize=True)

            # img2img
            image = 'test.jpg'
            images = ['test1.jpg', 'test2.jpg']

            # predict one
            image = process.single_predict(prompt, images=image, is_visualize=True)

            # predict batch
            images = process.batch_predict(prompts, neg_texts=neg_prompts, images=image, batch_size=2, is_visualize=True)     # base on same image
            images = process.batch_predict(prompts, neg_texts=neg_prompts, images=images, batch_size=2, is_visualize=True)    # base on different image
    """


class SimpleTextImage(DataProcess):
    dataset_version = 'simple_text_image'
    data_dir = 'data/simple_text_image'
    train_data_num = 40000  # do not set too large, 'cause images will be cached in memory
    val_data_num = 512
    input_size = 512
    in_ch = 3

    def get_train_data(self, *args, task='images', text_task='texts', **kwargs):
        from data_parse.cv_data_parse.datasets.SimpleTextImage import Loader

        loader = Loader(self.data_dir, image_suffix='.png')
        iter_data = loader.load(
            generator=False,
            max_size=self.train_data_num,
            task=task,
            text_task=text_task
        )[0]
        return iter_data


class SD_SimpleTextImage(SD, SimpleTextImage):
    """
    Usage:
        .. code-block:: python

            from bundles.image_generation import SD_SimpleTextImage as Process

            # lora finetune
            process = Process(
                data_dir='xxx',

                use_half_lora=True,
                use_lora=True,

                model_config=dict(
                    sampler_config=dict(schedule_config=dict(
                        num_steps=20
                    )),
                    vae_config=dict(
                        attn_type=2,
                    ),
                    backbone_config=dict(
                        attend_type='ScaleAttendWithXformers',
                    )
                ),

                lora_config=dict(
                    r=128,
                    alpha=64
                ),

                vocab_fn='xxx/vocab.json',
                encoder_fn='xxx/merges.txt',

                config_version = 'v1.5',
                pretrained_model='xxx',
            )

            process.run(
                max_epoch=50, train_batch_size=16,
                use_scheduler=True,
                fit_kwargs=dict(check_period=1000, max_save_weight_num=10),
                metric_kwargs=dict(is_visualize=True),
            )
    """


class WithFluxLora(WithLora):
    def set_lora(self):
        assert 'di' in self.config_version, 'Only support `diffusers` version weights, for example, try to set config_version to `dev.di`'
        from models.tuning import lora

        self.lora_wrap = lora.ModelWrap(
            include=(
                'clip',
                'double_blocks',
                'single_blocks',
            ),
            exclude=(
                't5',
                'vae',
                'norm',
                'attend',
                'embedding',
                'act',
                'dropout',
                'attn_res',
                nn.Identity
            ),
            **self.lora_config
        )
        self.lora_wrap.wrap(self.model)

        for full_name in self.lora_wrap.layers:
            layer = torch_utils.ModuleManager.get_module_by_name(self.model, full_name)
            layer.to(self.device)
            if self.use_half_lora:
                layer.to(torch.bfloat16)

        self.register_save_checkpoint(self.save_lora_weight)
        self.log('Successfully add lora!')

    def load_lora_pretrain(self):
        if hasattr(self, 'lora_pretrained_model'):
            from models.image_generation.flux import WeightConverter
            from models.bundles import WeightLoader

            state_dict = WeightLoader.auto_load(self.lora_pretrained_model)
            state_dict = WeightConverter.from_official_lora(state_dict)
            self.lora_wrap.load_state_dict(state_dict, strict=True)
            self.log(f'Loaded lora pretrain model from {self.lora_pretrained_model}!')


class FromFluxPretrained(CheckpointHooks):
    clip_text_encoder_pretrained: List[str] | str
    t5_text_encoder_pretrained: List[str] | str
    flux_pretrained: List[str] | str
    vae_pretrained: List[str] | str

    def load_pretrained(self):
        from models.image_generation.flux import WeightConverter
        from models.bundles import WeightLoader

        state_dicts = {
            "t5": WeightLoader.auto_load(self.t5_text_encoder_pretrained, suffix='.safetensors'),
            "clip": WeightLoader.auto_load(self.clip_text_encoder_pretrained, suffix='.safetensors'),
            "flux": WeightLoader.auto_load(self.flux_pretrained, suffix='.safetensors'),
            "vae": WeightLoader.auto_load(self.vae_pretrained, suffix='.safetensors'),
        }

        state_dict = WeightConverter.auto_convert(state_dicts)
        self.model.load_state_dict(state_dict, strict=False, assign=True)

        self.log(f'Loaded pretrain model!')


class BaseFlux(DiProcess):
    model_version = 'flux'
    config_version: str = 'dev'

    in_ch = 3
    input_size = 768

    model_config: dict = {}

    def set_model(self):
        from models.image_generation.flux import Model, Config

        model_config = configs.ConfigObjParse.merge_dict(Config.get(self.config_version), self.model_config)
        with torch.device('meta'):  # fast to init model
            self.model = Model(**model_config)

    clip_tokenizer: 'CLIPTokenizer'
    clip_vocab_fn: str
    clip_encoder_fn: str

    t5_tokenizer: 'T5Tokenizer'
    t5_vocab_fn: str
    t5_encoder_fn: str

    def set_tokenizer(self):
        from data_parse.nl_data_parse.pre_process.bundled import CLIPTokenizer, T5Tokenizer

        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            self.clip_vocab_fn,
            self.clip_encoder_fn,
            max_seq_len=77
        )

        self.t5_tokenizer = T5Tokenizer.from_pretrained(
            self.t5_vocab_fn,
            self.t5_encoder_fn,
            max_seq_len=512
        )

    def get_model_inputs(self, loop_inputs, train=True):
        if train:
            raise NotImplementedError
        else:
            return self.get_model_val_inputs(loop_inputs)


class FluxPredictor(BaseFlux):
    val_aug = Apply([
        scale.Rectangle(),
        channel.BGR2RGB(),
        channel.HWC2CHW(),
    ])

    def val_data_augment(self, ret) -> dict:
        if 'image' in ret and ret['image'] is not None:
            ret.update(dst=self.input_size)
            ret.update(self.val_aug(**ret))
        return ret

    def get_model_val_inputs(self, loop_inputs):
        texts = []
        images = []
        mask_images = []
        for ret in loop_inputs:
            if 'text' in ret:
                texts.append(ret['text'])

            if 'image' in ret and ret['image'] is not None:
                image = ret.pop('image')
                if image.shape[0] == 4:
                    image, mask_image = image[:-1], image[-1:]
                    if 'mask_image' not in ret or ret['mask_image'] is None:
                        mask_images.append(torch.from_numpy(mask_image).to(self.device, non_blocking=True, dtype=torch.float))

                images.append(torch.from_numpy(image).to(self.device, non_blocking=True, dtype=torch.float))

        inputs = self.clip_tokenizer.encode_paragraphs(texts)
        clip_text_ids = torch.tensor(inputs['segments_ids']).to(self.device)

        inputs = self.t5_tokenizer.encode_paragraphs(texts)
        t5_text_ids = torch.tensor(inputs['segments_ids']).to(self.device)

        if images:
            images = torch.stack(images)
            images /= 255.
            images = images * 2 - 1  # normalize, [0, 1] -> [-1, 1]

        if mask_images:
            mask_images = torch.stack(mask_images)
            mask_images /= 255

        return dict(
            clip_text_ids=clip_text_ids,
            t5_text_ids=t5_text_ids,
            x=images,
            image_size=self.input_size
        )

    def gen_predict_inputs(self, *objs, images=None, start_idx=None, end_idx=None, **kwargs):
        """

        Args:
            *objs:
            images (str|List[str]|np.ndarray|List[np.ndarray]):
            start_idx:
            end_idx:
            **kwargs:

        Returns:

        """
        pos_texts = objs[0][start_idx: end_idx]
        b = len(pos_texts)

        images = _load_images(images, b, start_idx, end_idx)

        rets = []
        for text, image in zip(pos_texts, images):
            rets.append(dict(
                image=image,
                text=text,
                **kwargs
            ))

        return rets


class Flux(WithFluxLora, FromFluxPretrained, FluxPredictor):
    """no training, only for prediction

    Usage:
        .. code-block:: python

            from bundles.image_generation import Flux as Process

            model_dir = 'xxx'
            process = Process(
                clip_vocab_fn=f'{model_dir}/tokenizer/vocab.json',
                clip_encoder_fn=f'{model_dir}/tokenizer/merges.txt',

                t5_vocab_fn=f'{model_dir}/tokenizer_2/tokenizer.json',
                t5_encoder_fn=f'{model_dir}/tokenizer_2/spiece.model',

                clip_text_encoder_pretrained=f'{model_dir}/text_encoder/model.safetensors',
                t5_text_encoder_pretrained=f'{model_dir}/text_encoder_2',
                flux_pretrained=f'{model_dir}/flux1-dev.safetensors',
                vae_pretrained=f'{model_dir}/ae.safetensors',

                low_memory_run=True,
                use_half=True,

                # if using `diffusers` version weights
                # config_version='dev.di',
                # flux_pretrained=f'{model_dir}/transformer',

                # if using lora
                # use_lora=True,
                # lora_pretrained_model='xxx',
                # lora_config=dict(
                #     r=16,
                #     alpha=16
                # )
            )
            process.init()

            # txt2img
            # predict one
            prompt = 'a painting of a virus monster playing guitar'
            image = process.single_predict(
                prompt,
                is_visualize=True,
                model_kwargs=dict(
                    image_size=1024,
                    num_steps=20,
                )
            )

            # predict batch
            prompts = [prompt] * 2
            images = process.batch_predict(prompts, batch_size=4, is_visualize=True)
    """
