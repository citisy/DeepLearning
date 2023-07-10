import os
from .base import Process, BaseDataset
import math
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import os_lib, converter, configs
from utils.visualize import ImageVisualize
from utils.torch_utils import EarlyStopping, ModuleInfo, Export
from utils.os_lib import MemoryInfo
from data_parse.cv_data_parse.base import DataRegister, DataVisualizer
from data_parse.cv_data_parse.data_augmentation import channel
import copy
import cv2
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply, Apply, channel


class IgDataset(BaseDataset):
    pass


class WGAN_Mnist(Process):
    dataset = IgDataset

    def __init__(self, device=0):
        from models.image_generate.wgan import Model

        input_size = 64
        in_ch = 3
        hidden_ch = 100

        super().__init__(
            model=Model(
                input_size=input_size,
                in_ch=in_ch,
                hidden_ch=hidden_ch,
            ),
            model_version='WGAN',
            # dataset_version='mnist',
            dataset_version='fashion',
            input_size=input_size,
            device=device
        )

    def model_info(self, depth=4):
        profile = ModuleInfo.profile_per_layer(self.model.model_d, depth=depth)
        s = f'net d module info: \n{"name":<20}{"module":<40}{"params":>10}{"grads":>10}\n'

        for p in profile:
            s += f'{p[0]:<20}{p[1]:<40}{p[2]["params"]:>10}{p[2]["grads"]:>10}\n'

        self.logger.info(s)

        profile = ModuleInfo.profile_per_layer(self.model.model_g, depth=depth)
        s = f'net g module info: \n{"name":<20}{"module":<40}{"params":>10}{"grads":>10}\n'

        for p in profile:
            s += f'{p[0]:<20}{p[1]:<40}{p[2]["params"]:>10}{p[2]["grads"]:>10}\n'

        self.logger.info(s)

    def get_train_data(self):
        from data_parse.cv_data_parse.Mnist import Loader

        # loader = Loader(f'data/mnist')
        loader = Loader(f'data/fashion')
        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False)[0]

        return data

    def data_augment(self, ret):
        ret.update(dst=self.input_size)
        ret.update(Apply([
            channel.Gray2BGR(),
            scale.Proportion(),
            pixel_perturbation.MinMax(),
            pixel_perturbation.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            channel.HWC2CHW()
        ])(**ret))

        return ret

    def fit(self, dataset, max_epoch, batch_size, save_period=None, **dataloader_kwargs):
        # sampler = distributed.DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset,
            shuffle=True,
            # sampler=sampler,
            pin_memory=True,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            **dataloader_kwargs
        )

        self.model.to(self.device)
        model_g = self.model.model_g
        model_d = self.model.model_d

        optimizer_d = optim.Adam(model_d.parameters(), lr=0.00005, betas=(0.5, 0.999))
        optimizer_g = optim.Adam(model_g.parameters(), lr=0.00005, betas=(0.5, 0.999))

        val_noise = torch.normal(mean=0., std=1., size=(batch_size, self.model.hidden_ch, 1, 1), device=self.device)
        flag = True

        gen_iter = 0
        j = 0
        for i in range(max_epoch):
            model_g.train()
            model_d.train()
            pbar = tqdm(dataloader, desc=f'train {i}/{max_epoch}')

            for rets in pbar:
                images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images = torch.stack(images)

                model_d.requires_grad_(True)

                # see also https://github.com/martinarjovsky/WassersteinGAN/issues/18
                for p in model_d.parameters():
                    p.data.clamp_(-0.01, 0.01)

                model_d.zero_grad()

                real_y = model_d(images)

                # see also https://github.com/martinarjovsky/WassersteinGAN/issues/9
                loss_d_real = real_y
                loss_d_real.backward()

                with torch.no_grad():   # do not train g
                    noise = torch.normal(mean=0., std=1., size=(batch_size, self.model.hidden_ch, 1, 1), device=self.device)
                    fake_x = model_g(noise)

                fake_y = model_d(fake_x)
                loss_d_fake = -fake_y
                loss_d_fake.backward()
                optimizer_d.step()

                if gen_iter < 25 or gen_iter % 500 == 0:
                    iter_gap = 100
                else:
                    iter_gap = 5

                if j % iter_gap == iter_gap - 1:
                    model_d.requires_grad_(False)  # do not train d
                    model_g.zero_grad()
                    noise = torch.normal(mean=0., std=1., size=(batch_size, self.model.hidden_ch, 1, 1), device=self.device)
                    fake_x = model_g(noise)
                    loss_g = model_d(fake_x)
                    loss_g.backward()
                    optimizer_g.step()

                    pbar.set_postfix({
                        'gen_iter': gen_iter,
                        'loss_d': f'{(real_y - fake_y).item():.06}',
                        'loss_g': f'{loss_g.item():.06}',
                        'loss_d_real': f'{loss_d_real.item():.06}',
                        'loss_d_fake': f'{loss_d_fake.item():.06}',
                        # 'cpu_info': MemoryInfo.get_process_mem_info(),
                        # 'gpu_info': MemoryInfo.get_gpu_mem_info()
                    })

                    gen_iter += 1

                    if save_period and gen_iter % save_period == 0:
                        if flag:
                            images = images.mul(0.5).add(0.5).mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

                            ret = [[{'image': image, '_id': 'real.png'}] for image in images]
                            DataVisualizer(f'cache_data/{self.dataset_version}', verbose=True, stdout_method=self.logger.info)(*ret)
                            flag = False

                        self.metric(val_noise, gen_iter)
                        # self.save(f'{self.model_dir}/{self.dataset_version}_{gen_iter}.pth')

                j += 1

    def metric(self, x, gen_iter):
        with torch.no_grad():
            fake_y = self.model.model_g(x)
            fake_y = fake_y.data.mul(0.5).add(0.5).mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            ret = [[{'image': image, '_id': f'{gen_iter}_fake.png'}] for image in fake_y]
            DataVisualizer(f'cache_data/{self.dataset_version}', verbose=True, stdout_method=self.logger.info)(*ret)

