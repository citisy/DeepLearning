from .base import Process, BaseDataset, WEIGHT
import random
import itertools
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.torch_utils import EarlyStopping, ModuleInfo, Export
from utils import os_lib, configs
from data_parse.cv_data_parse.base import DataRegister, DataVisualizer
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply, Apply, channel
from pathlib import Path
import numpy as np
from datetime import datetime


class IgOptimizer:
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
    dataset = BaseDataset
    total_nums = 0

    def model_info(self, depth=None, **kwargs):
        modules = dict(
            d=self.model.net_d,
            g=self.model.net_g,
        )

        for key, module in modules.items():
            profile = ModuleInfo.profile_per_layer(module, depth=depth)
            s = f'net {key} module info: \n{"name":<20}{"module":<40}{"params":>10}{"grads":>10}\n'

            for p in profile:
                s += f'{p[0]:<20}{p[1]:<40}{p[2]["params"]:>10}{p[2]["grads"]:>10}\n'

            self.logger.info(s)

    def on_train_epoch_end(self, total_nums, save_period, val_dataloader, train_batch_size,
                           mean_loss=None, max_size=None, **metric_kwargs):
        mean_loss_g, mean_loss_d = mean_loss
        if save_period and total_nums % save_period < train_batch_size:
            self.wandb.log_info = {'total_nums': total_nums, 'mean_loss_g': mean_loss_g, 'mean_loss_d': mean_loss_d}
            self.metric(val_dataloader, cur_epoch=total_nums, **metric_kwargs)

            ckpt = {
                'total_nums': total_nums,
                'wandb_id': self.wandb_run.id,
                'date': datetime.now().isoformat()
            }

            self.save(f'{self.work_dir}/{total_nums}.pth', save_type=WEIGHT, **ckpt)
            os_lib.FileCacher(f'{self.work_dir}/', max_size=max_size, stdout_method=self.logger.info).delete_over_range(suffix='pth')

            self.wandb.log(self.wandb.log_info)

    def metric(self, *args, **kwargs):
        self.predict(*args, **kwargs)

    def on_val_step_end(self, vis_image, _ids, cur_epoch, visualize, batch_size, max_vis_num, vis_num):
        if visualize:
            n = min(batch_size, max_vis_num - vis_num)
            if n > 0:
                ret = []
                for name, images in vis_image.items():
                    images = images.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
                    ret.append([{'image': image, '_id': _id} for image, _id in zip(images, _ids)])

                ret = [r[:n] for r in ret]
                cache_dir = f'{self.save_result_dir}/{cur_epoch}'
                cache_image = DataVisualizer(cache_dir, verbose=False, pbar=False)(*ret, return_image=True)
                self.log_info.setdefault('val_image', []).extend([self.wandb.Image(img, caption=Path(_id).stem) for img, _id in zip(cache_image, _ids)])
                vis_num += n
        return vis_num


class Mnist(Process):
    def get_train_data(self):
        from data_parse.cv_data_parse.Mnist import Loader

        # loader = Loader(f'data/mnist')
        loader = Loader(f'data/fashion')
        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False)[0]

        return data

    aug = Apply([
        channel.Gray2BGR(),
        scale.Proportion(),
        pixel_perturbation.MinMax(),
        pixel_perturbation.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))

        return ret

    def get_val_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.Mnist import Loader

        # loader = Loader(f'data/mnist')
        loader = Loader(f'data/fashion')
        data = loader(set_type=DataRegister.TEST, image_type=DataRegister.ARRAY, generator=False)[0]

        return data

    def val_data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))

        return ret


class WGAN(IgProcess):
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
            optimizer=IgOptimizer(optimizer_d, optimizer_g),
            model_version=model_version,
            input_size=input_size,
            **kwargs
        )

    def fit(self, max_epoch, batch_size, save_period=None, save_maxsize=None, metric_kwargs=dict(), **dataloader_kwargs):
        train_dataloader, val_dataloader, metric_kwargs = self.on_train_start(batch_size, metric_kwargs, **dataloader_kwargs)

        self.model.to(self.device)
        optimizer_d, optimizer_g = self.optimizer.optimizer_d, self.optimizer.optimizer_g

        val_noise = torch.normal(mean=0., std=1., size=(64, self.model.hidden_ch, 1, 1), device=self.device)
        if save_period:
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

                    pbar.set_postfix({
                        'total_nums': self.total_nums,
                        'mean_loss_d': f'{mean_loss_d:.06}',
                        'mean_loss_g': f'{mean_loss_g:.06}',
                        # 'cpu_info': MemoryInfo.get_process_mem_info(),
                        # 'gpu_info': MemoryInfo.get_gpu_mem_info()
                    })

                    if self.on_train_epoch_end(self.total_nums, save_period, val_noise, batch_size,
                                               mean_loss=(mean_loss_g, mean_loss_d), save_maxsize=save_maxsize,
                                               **metric_kwargs):
                        break

    def predict(self, val_noise, batch_size=128, cur_epoch=-1, model=None, visualize=False, max_vis_num=None, save_ret_func=None, **dataloader_kwargs):
        self.model.to(self.device)
        max_vis_num = max_vis_num or float('inf')
        vis_num = 0

        with torch.no_grad():
            for i in tqdm(range(0, len(val_noise), batch_size), desc='val'):
                x = val_noise[i:i + batch_size].to(self.device)
                fake_y = self.model.net_g(x)
                vis_image = dict(
                    fake=fake_y,
                )

                vis_num = self.on_val_step_end(vis_image, cur_epoch, visualize, batch_size, max_vis_num, vis_num)

    def on_val_step_end(self, vis_image, cur_epoch, visualize, batch_size, max_vis_num, vis_num):
        if visualize:
            n = min(batch_size, max_vis_num - vis_num)
            if n > 0:
                ret = []
                for name, images in vis_image.items():
                    images = images.data.mul(0.5).add(0.5).mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
                    ret.append([{'image': image, '_id': f'{name}.png'} for image in images])

                ret = [r for r in zip(*ret)]
                cache_dir = f'{self.save_result_dir}/{cur_epoch}'
                DataVisualizer(cache_dir, verbose=False, stdout_method=self.logger.info)(*ret[:n])
                self.log_info['val_image'] = [self.wandb.Image(str(fp), caption=fp.stem) for fp in Path(cache_dir).glob('*.png')]
                vis_num += n
        return vis_num


class WGAN_Mnist(WGAN, Mnist):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import WGAN_Mnist as Process

            Process().run(max_epoch=1000, train_batch_size=64, save_period=10000, metric_kwargs=dict(visualize=True, max_vis_num=64))
    """

    def __init__(self, dataset_version='fashion', **kwargs):
        super().__init__(dataset_version=dataset_version, **kwargs)


class Facade(Process):
    def get_train_data(self):
        from data_parse.cv_data_parse.cmp_facade import Loader

        loader = Loader(f'data/cmp_facade')
        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False)[0]

        return data

    aug = Apply([
        scale.LetterBox(),
        pixel_perturbation.MinMax(),
        # pixel_perturbation.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))
        pix_image = ret['pix_image']
        pix_image = self.aug.apply_image(pix_image, ret)
        ret['pix_image'] = pix_image

        return ret

    def get_val_data(self, *args, **kwargs):
        from data_parse.cv_data_parse.cmp_facade import Loader

        loader = Loader(f'data/cmp_facade')
        data = loader(set_type=DataRegister.TEST, image_type=DataRegister.ARRAY, generator=False)[0]

        return data

    def val_data_augment(self, ret) -> dict:
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))
        pix_image = ret['pix_image']
        pix_image = self.aug.apply_image(pix_image, ret)
        ret['pix_image'] = pix_image

        return ret

    def val_data_restore(self, ret) -> dict:
        ret = self.aug.restore(ret)
        return ret


class Pix2pix(IgProcess):
    def __init__(self,
                 model_version='Pix2pix',
                 input_size=256,
                 in_ch=3,
                 **kwargs
                 ):
        from models.image_generate.pix2pix import Model

        model = Model(
            in_ch=in_ch,
            input_size=input_size
        )

        optimizer_g = optim.Adam(model.net_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(model.net_d.parameters(), lr=0.0002, betas=(0.5, 0.999))

        super().__init__(
            model=model,
            optimizer=IgOptimizer(optimizer_d, optimizer_g),
            model_version=model_version,
            input_size=input_size,
            **kwargs
        )

    def fit(self, max_epoch, batch_size, save_period=None, save_maxsize=None, metric_kwargs=dict(), **dataloader_kwargs):
        train_dataloader, val_dataloader, metric_kwargs = self.on_train_start(batch_size, metric_kwargs, **dataloader_kwargs)

        optimizer_d, optimizer_g = self.optimizer.optimizer_d, self.optimizer.optimizer_g

        for i in range(max_epoch):
            self.model.train()
            pbar = tqdm(train_dataloader, desc=f'train {i}/{max_epoch}')
            total_nums = 0
            total_loss_g = 0
            total_loss_d = 0

            for rets in pbar:
                images_a = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_a = torch.stack(images_a)

                images_b = [torch.from_numpy(ret.pop('pix_image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_b = torch.stack(images_b)

                real_a = images_a
                real_b = images_b
                fake_b = self.model.net_g(real_a)
                fake_ab = torch.cat((real_a, fake_b), 1)

                optimizer_d.zero_grad()
                loss_d = self.model.loss_d(real_a, real_b, fake_ab)
                loss_d.backward()
                optimizer_d.step()

                optimizer_g.zero_grad()
                loss_g = self.model.loss_g(real_b, fake_b, fake_ab)
                loss_g.backward()
                optimizer_g.step()

                total_loss_g += loss_g.item()
                total_loss_d += loss_d.item()

                total_nums += len(rets)
                self.total_nums += len(rets)
                mean_loss_g = total_loss_g / total_nums
                mean_loss_d = total_loss_d / total_nums

                pbar.set_postfix({
                    'total_nums': self.total_nums,
                    'loss_g': f'{loss_g.item():.06}',
                    'loss_d': f'{loss_d.item():.06}',
                    'mean_loss_g': f'{mean_loss_g:.06}',
                    'mean_loss_d': f'{mean_loss_d:.06}',
                    # 'cpu_info': MemoryInfo.get_process_mem_info(),
                    # 'gpu_info': MemoryInfo.get_gpu_mem_info()
                })

                if self.on_train_epoch_end(self.total_nums, save_period, val_dataloader, batch_size,
                                           mean_loss=(mean_loss_g, mean_loss_d), save_maxsize=save_maxsize,
                                           **metric_kwargs):
                    break

    def predict(self, val_dataloader=None, batch_size=16, cur_epoch=-1, model=None, visualize=False, max_vis_num=None, save_ret_func=None, **dataloader_kwargs):
        if val_dataloader is None:
            val_dataloader = self.on_val_start(batch_size, **dataloader_kwargs)

        self.model.to(self.device)
        max_vis_num = max_vis_num or float('inf')
        vis_num = 0

        with torch.no_grad():
            for rets in tqdm(val_dataloader, desc='val'):
                images_a = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_a = torch.stack(images_a)

                images_b = [torch.from_numpy(ret.pop('pix_image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_b = torch.stack(images_b)

                real_a = images_a
                real_b = images_b
                fake_b = self.model.net_g(real_a)

                vis_image = dict(
                    real_a=real_a,
                    real_b=real_b,
                    fake_b=fake_b
                )
                _ids = [r['_id'] for r in rets]

                vis_num = self.on_val_step_end(vis_image, _ids, cur_epoch, visualize, batch_size, max_vis_num, vis_num)


class Pix2pix_facade(Pix2pix, Facade):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import Pix2pix_facade as Process

            Process().run(max_epoch=1000, train_batch_size=16, save_period=2000)
    """

    def __init__(self, dataset_version='facade', **kwargs):
        super().__init__(dataset_version=dataset_version, **kwargs)


class CycleGan(IgProcess):
    def __init__(self,
                 model_version='CycleGan',
                 input_size=256,
                 in_ch=3,
                 **kwargs
                 ):
        from models.image_generate.CycleGan import Model
        model = Model(
            in_ch=in_ch,
            input_size=input_size
        )

        optimizer_g = optim.Adam(itertools.chain(model.net_g_a.parameters(), model.net_g_b.parameters()), lr=0.00005, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(itertools.chain(model.net_d_a.parameters(), model.net_d_b.parameters()), lr=0.00005, betas=(0.5, 0.999))

        super().__init__(
            model=model,
            optimizer=(optimizer_d, optimizer_g),
            model_version=model_version,
            input_size=input_size,
            **kwargs
        )

    def model_info(self, depth=None, **kwargs):
        modules = dict(
            d_a=self.model.net_d_a,
            d_b=self.model.net_d_b,
            g_a=self.model.net_g_a,
            g_b=self.model.net_g_b
        )

        for key, module in modules.items():
            profile = ModuleInfo.profile_per_layer(module, depth=depth)
            s = f'net {key} module info: \n{"name":<20}{"module":<40}{"params":>10}{"grads":>10}\n'

            for p in profile:
                s += f'{p[0]:<20}{p[1]:<40}{p[2]["params"]:>10}{p[2]["grads"]:>10}\n'

            self.logger.info(s)

    def fit(self, max_epoch, batch_size, save_period=None, save_maxsize=None, metric_kwargs=dict(), **dataloader_kwargs):
        train_dataloader, val_dataloader, metric_kwargs = self.on_train_start(batch_size, metric_kwargs, **dataloader_kwargs)

        optimizer_d, optimizer_g = self.optimizer

        net_g_a = self.model.net_g_a
        net_g_b = self.model.net_g_b
        net_d_a = self.model.net_d_a
        net_d_b = self.model.net_d_b

        fake_a_cacher = os_lib.MemoryCacher(max_size=50)
        fake_b_cacher = os_lib.MemoryCacher(max_size=50)

        lambda_a = 10
        lambda_b = 10

        for i in range(max_epoch):
            self.model.train()
            pbar = tqdm(train_dataloader, desc=f'train {i}/{max_epoch}')
            total_nums = 0
            total_loss_g = 0
            total_loss_d = 0

            for rets in pbar:
                images_a = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_a = torch.stack(images_a)

                images_b = [torch.from_numpy(ret.pop('pix_image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_b = torch.stack(images_b)

                real_a = images_a.to(self.device)
                real_b = images_b.to(self.device)

                optimizer_g.zero_grad()
                (fake_b, rec_a), (loss_idt_b, loss_g_a, loss_cycle_a) = self.model.loss_g(real_a, net_g_a, net_g_b, net_d_a, lambda_a)
                (fake_a, rec_b), (loss_idt_a, loss_g_b, loss_cycle_b) = self.model.loss_g(real_b, net_g_b, net_g_a, net_d_b, lambda_b)
                loss_g = loss_g_a + loss_g_b + loss_cycle_a + loss_cycle_b + loss_idt_a + loss_idt_b
                loss_g.backward()
                optimizer_g.step()

                optimizer_d.zero_grad()
                loss_d_a = self.model.loss_d(real_a, fake_a, net_d_b, fake_a_cacher)
                loss_d_b = self.model.loss_d(real_b, fake_b, net_d_a, fake_b_cacher)
                loss_d = loss_d_a + loss_d_b
                loss_d.backward()
                optimizer_d.step()

                total_nums += len(rets)
                self.total_nums += len(rets)

                total_loss_g += loss_g.item()
                total_loss_d += loss_d.item()

                mean_loss_g = total_loss_g / total_nums
                mean_loss_d = total_loss_d / total_nums

                pbar.set_postfix({
                    'total_nums': self.total_nums,
                    'loss_g': f'{loss_g.item():.06}',
                    'loss_d': f'{loss_d.item():.06}',
                    'mean_loss_g': f'{mean_loss_g:.06}',
                    'mean_loss_d': f'{mean_loss_d:.06}',
                    # 'cpu_info': MemoryInfo.get_process_mem_info(),
                    # 'gpu_info': MemoryInfo.get_gpu_mem_info()
                })

                if self.on_train_epoch_end(self.total_nums, save_period, val_dataloader, batch_size,
                                           mean_loss=(mean_loss_g, mean_loss_d), save_maxsize=save_maxsize,
                                           **metric_kwargs):
                    break

    def predict(self, val_dataloader=None, batch_size=16, cur_epoch=-1, model=None, visualize=False, max_vis_num=None, save_ret_func=None, **dataloader_kwargs):
        if val_dataloader is None:
            val_dataloader = self.on_val_start(batch_size, **dataloader_kwargs)

        self.model.to(self.device)
        max_vis_num = max_vis_num or float('inf')
        vis_num = 0

        with torch.no_grad():
            for rets in tqdm(val_dataloader, desc='val'):
                images_a = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_a = torch.stack(images_a)

                images_b = [torch.from_numpy(ret.pop('pix_image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_b = torch.stack(images_b)

                real_a = images_a
                real_b = images_b

                fake_b = self.model.net_g_a(real_a)
                fake_a = self.model.net_g_b(real_b)

                rec_a = self.model.net_g_b(fake_b)
                rec_b = self.model.net_g_a(fake_a)

                vis_image = dict(
                    real_a=real_a,
                    fake_a=fake_a,
                    rec_a=rec_a,
                    real_b=real_b,
                    fake_b=fake_b,
                    rec_b=rec_b
                )
                _ids = [r['_id'] for r in rets]

                vis_num = self.on_val_step_end(vis_image, _ids, cur_epoch, visualize, batch_size, max_vis_num, vis_num)

    def batch_predict(self, images, batch_size=16, **kwargs):
        results = []
        with torch.no_grad():
            self.model.eval()
            for i in range(0, len(images), batch_size):
                rets = [self.val_data_augment({'image': image}) for image in images[i:i + batch_size]]
                images = [torch.from_numpy(ret.pop('image')).to(self.device) for ret in rets]
                images = torch.stack(images)

                outputs = self.model.net_g_b(images)
                outputs = [{k: v.to('cpu').numpy() for k, v in t.items()} for t in outputs]

                for ret, output in zip(rets, outputs):
                    output = configs.merge_dict(ret, output)
                    output = self.val_data_restore(output)
                    results.append(output)

        return results


class CycleGan_facade(CycleGan, Facade):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import CycleGan_facade as Process

            Process().run(max_epoch=1000, train_batch_size=8, save_period=500)
    """

    def __init__(self, dataset_version='facade', **kwargs):
        super().__init__(dataset_version=dataset_version, **kwargs)
