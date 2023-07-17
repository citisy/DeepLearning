from .base import Process, BaseDataset
import random
import itertools
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.torch_utils import EarlyStopping, ModuleInfo, Export
from utils import visualize, os_lib
from data_parse.cv_data_parse.base import DataRegister, DataVisualizer
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply, Apply, channel


class IgProcess(Process):
    dataset = BaseDataset

    def model_info(self, depth=None):
        modules = dict(
            d=self.model.net_d,
            g=self.model.net_g,
        )

        for key, module in modules:
            profile = ModuleInfo.profile_per_layer(module, depth=depth)
            s = f'net {key} module info: \n{"name":<20}{"module":<40}{"params":>10}{"grads":>10}\n'

            for p in profile:
                s += f'{p[0]:<20}{p[1]:<40}{p[2]["params"]:>10}{p[2]["grads"]:>10}\n'

            self.logger.info(s)

    def run(self, max_epoch=2000, train_batch_size=64, predict_batch_size=None, save_period=None):
        ####### train ###########
        data = self.get_train_data()

        dataset = self.dataset(
            data,
            augment_func=self.data_augment,
            complex_augment_func=self.complex_data_augment if hasattr(self, 'complex_data_augment') else None
        )

        self.fit(
            dataset, max_epoch, train_batch_size, save_period,
            num_workers=16
        )
        self.save(self.model_path)


class WGAN_Mnist(IgProcess):
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
        net_g = self.model.net_g
        net_d = self.model.net_d

        optimizer_d = optim.Adam(net_d.parameters(), lr=0.00005, betas=(0.5, 0.999))
        optimizer_g = optim.Adam(net_g.parameters(), lr=0.00005, betas=(0.5, 0.999))

        val_noise = torch.normal(mean=0., std=1., size=(batch_size, self.model.hidden_ch, 1, 1), device=self.device)
        flag = True

        gen_iter = 0
        j = 0
        for i in range(max_epoch):
            net_g.train()
            net_d.train()
            pbar = tqdm(dataloader, desc=f'train {i}/{max_epoch}')

            for rets in pbar:
                images = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images = torch.stack(images)

                # update D
                net_d.requires_grad_(True)

                # note that, weight clipping, if change to gradient penalty, it is WGAN-GP
                # see also https://github.com/martinarjovsky/WassersteinGAN/issues/18
                for p in net_d.parameters():
                    p.data.clamp_(-0.01, 0.01)

                net_d.zero_grad()

                # note that, pred output backward without loss function,
                # see also https://github.com/martinarjovsky/WassersteinGAN/issues/9
                # 1. real_x -> net_d -> pred_real -> loss_d_real -> gradient_descent
                # loss_d_real = net_d(real_x)
                real_x = images
                pred_real = net_d(real_x)
                loss_d_real = pred_real
                loss_d_real.backward()

                # 2. noise -> net_g -> fake_x -> net_d -> pred_fake -> loss_d_fake -> gradient_ascent
                # loss_d_fake = net_d(net_g(noise))
                with torch.no_grad():
                    noise = torch.normal(mean=0., std=1., size=(batch_size, self.model.hidden_ch, 1, 1), device=self.device)
                    fake_x = net_g(noise)

                pred_fake = net_d(fake_x)
                loss_d_fake = -pred_fake
                loss_d_fake.backward()
                optimizer_d.step()

                if gen_iter < 25 or gen_iter % 500 == 0:
                    iter_gap = 100
                else:
                    iter_gap = 5

                if j % iter_gap == iter_gap - 1:
                    net_d.requires_grad_(False)
                    net_g.zero_grad()

                    # 3. noise -> net_g -> fake_x -> net_d -> pred_fake -> loss_g -> gradient_descent
                    # loss_g = net_d(net_g(noise))
                    noise = torch.normal(mean=0., std=1., size=(batch_size, self.model.hidden_ch, 1, 1), device=self.device)
                    fake_x = net_g(noise)
                    loss_g = net_d(fake_x)
                    loss_g.backward()
                    optimizer_g.step()

                    pbar.set_postfix({
                        'gen_iter': gen_iter,
                        'loss_d': f'{(pred_real - pred_fake).item():.06}',
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
            fake_y = self.model.net_g(x)
            fake_y = fake_y.data.mul(0.5).add(0.5).mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
            ret = [[{'image': image, '_id': f'{gen_iter}_fake.png'}] for image in fake_y]
            DataVisualizer(f'cache_data/{self.dataset_version}', verbose=True, stdout_method=self.logger.info)(*ret)


def facade_data_aug(ret, input_size):
    ret.update(dst=input_size)
    ret.update(Apply([
        scale.LetterBox(),
        pixel_perturbation.MinMax(),
        # pixel_perturbation.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        channel.HWC2CHW()
    ])(**ret))

    pix_image = ret['pix_image']
    pix_image = scale.Proportion().apply_image(pix_image, **ret['scale.Proportion'])
    pix_image = crop.Pad().apply_image(pix_image, **ret['crop.Pad'])
    pix_image = crop.Crop().apply_image(pix_image, **ret['crop.Crop'])
    pix_image = pixel_perturbation.MinMax().apply_image(pix_image)
    pix_image = channel.HWC2CHW().apply_image(pix_image)

    ret['pix_image'] = pix_image

    return ret


class Pix2pix_facade(IgProcess):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import Pix2pix_facade as Process

            Process().run(max_epoch=2000, train_batch_size=16, save_period=2000)
    """

    def __init__(self, device=0):
        from models.image_generate.pix2pix import Model

        input_size = 256
        in_ch = 3

        super().__init__(
            model=Model(
                in_ch=in_ch,
                input_size=input_size
            ),
            model_version='Pix2pix',
            dataset_version='facade',
            input_size=input_size,
            device=device
        )

    def get_train_data(self):
        from data_parse.cv_data_parse.cmp_facade import Loader

        loader = Loader(f'data/cmp_facade')
        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False)[0]

        return data

    def data_augment(self, ret):
        return facade_data_aug(ret, self.input_size)

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
        net_g = self.model.net_g
        net_d = self.model.net_d
        optimizer_g = torch.optim.Adam(net_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(net_d.parameters(), lr=0.0002, betas=(0.5, 0.999))

        gan_loss_fn = nn.MSELoss()
        real_label = torch.tensor(1., device=self.device)
        fake_label = torch.tensor(0., device=self.device)
        l1_loss_fn = torch.nn.L1Loss()
        lambda_l1 = 100.0

        j = 0
        for i in range(max_epoch):
            self.model.train()
            pbar = tqdm(dataloader, desc=f'train {i}/{max_epoch}')

            for rets in pbar:
                images_a = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_a = torch.stack(images_a)

                images_b = [torch.from_numpy(ret.pop('pix_image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_b = torch.stack(images_b)

                real_a = images_a.to(self.device)
                real_b = images_b.to(self.device)

                fake_b = net_g(real_a)

                # update D
                net_d.requires_grad_(True)
                optimizer_d.zero_grad()

                # 2. real_a + fake_b -> net_d -> pred_fake, fake_label -> loss_d_fake
                # loss_d_fake = net_d(real_a, fake_b) - 0
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = net_d(fake_ab.detach())
                loss_d_fake = gan_loss_fn(pred_fake, fake_label.expand_as(pred_fake))

                # 3. real_a + real_b -> net_d -> pred_real, real_label -> loss_d_real
                # loss_d_real = net_d(real_a, real_b) - 1
                real_ab = torch.cat((real_a, real_b), 1)
                pred_real = net_d(real_ab)
                loss_d_real = gan_loss_fn(pred_real, real_label.expand_as(pred_real))

                loss_d = (loss_d_fake + loss_d_real) * 0.5
                loss_d.backward()
                optimizer_d.step()

                # update G
                net_d.requires_grad_(False)
                optimizer_g.zero_grad()

                #  4. real_a + fake_b -> net_d -> pred_fake, real_label -> loss_g_gan
                # loss_g_gan = net_d(real_a, fake_b)
                pred_fake = net_d(fake_ab)
                loss_g_gan = gan_loss_fn(pred_fake, real_label.expand_as(pred_fake))

                # 1. real_a -> net_g -> fake_b, real_b-> loss_g_l1
                # loss_g_l1 = ||net_g(real_a) - real_b||
                loss_g_l1 = l1_loss_fn(fake_b, real_b) * lambda_l1

                loss_g = loss_g_gan + loss_g_l1
                loss_g.backward()
                optimizer_g.step()

                pbar.set_postfix({
                    'gen_iter': j,
                    'loss_g_gan': f'{loss_g_gan.item():.06}',
                    'loss_g_l1': f'{loss_g_l1.item():.06}',
                    'loss_d_real': f'{loss_d_real.item():06}',
                    'loss_d_fake': f'{loss_d_fake.item():06}'
                    # 'cpu_info': MemoryInfo.get_process_mem_info(),
                    # 'gpu_info': MemoryInfo.get_gpu_mem_info()
                })

                j += 1
                if j % save_period == 0:
                    vis_image = dict(
                        real_a=real_a,
                        real_b=real_b,
                        fake_b=fake_b
                    )
                    ret = []
                    for name, images in vis_image.items():
                        images = images.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
                        ret.append([{'image': image, '_id': f'{j}_{name}.png'} for image in images])

                    ret = [r for r in zip(*ret)]
                    DataVisualizer(f'cache_data/{self.model_version}/{self.dataset_version}', verbose=True, stdout_method=self.logger.info)(*ret)


class CycleGan_facade(IgProcess):
    """
    Usage:
        .. code-block:: python

            from examples.image_generate import CycleGan_facade as Process

            Process().run(max_epoch=1000, train_batch_size=8, save_period=500)
    """

    def __init__(self, device=0):
        from models.image_generate.CycleGan import Model

        input_size = 256
        in_ch = 3

        super().__init__(
            model=Model(
                in_ch=in_ch,
                input_size=input_size
            ),
            model_version='CycleGan',
            dataset_version='facade',
            input_size=input_size,
            device=device
        )

    def model_info(self, depth=None):
        modules = dict(
            d_a=self.model.net_d_a,
            d_b=self.model.net_d_b,
            g_a=self.model.net_g_a,
            g_b=self.model.net_g_b
        )

        for key, module in modules:
            profile = ModuleInfo.profile_per_layer(module, depth=depth)
            s = f'net {key} module info: \n{"name":<20}{"module":<40}{"params":>10}{"grads":>10}\n'

            for p in profile:
                s += f'{p[0]:<20}{p[1]:<40}{p[2]["params"]:>10}{p[2]["grads"]:>10}\n'

            self.logger.info(s)

    def get_train_data(self):
        from data_parse.cv_data_parse.cmp_facade import Loader

        loader = Loader(f'data/cmp_facade')
        data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False)[0]

        return data

    def data_augment(self, ret):
        return facade_data_aug(ret, self.input_size)

    def fit(self, dataset, max_epoch, batch_size, save_period=None, **dataloader_kwargs):
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

        net_g_a = self.model.net_g_a
        net_g_b = self.model.net_g_b
        net_d_a = self.model.net_d_a
        net_d_b = self.model.net_d_b

        optimizer_g = torch.optim.Adam(itertools.chain(net_g_a.parameters(), net_g_b.parameters()), lr=0.00005, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(itertools.chain(net_d_a.parameters(), net_d_b.parameters()), lr=0.00005, betas=(0.5, 0.999))
        gan_loss_fn = nn.MSELoss()
        real_label = torch.tensor(1., device=self.device)
        fake_label = torch.tensor(0., device=self.device)
        cycle_l1_loss_fn = torch.nn.L1Loss()
        idt_l1_loss_fn = torch.nn.L1Loss()

        fake_a_cacher = os_lib.MemoryCacher(max_size=50)
        fake_b_cacher = os_lib.MemoryCacher(max_size=50)

        lambda_identity = 0.5
        lambda_a = 10
        lambda_b = 10

        def update_g(real_x, net_g_x, net_g_y, net_d_x, lambda_x):
            # symmetrically training
            # fake_x is compared with real_x and real_y, rec_x is compared with real_x
            net_d_x.requires_grad_(False)

            if lambda_identity > 0:
                # 1. real_x -> net_g_y -> idt_y, real_x -> loss_idt_y
                # loss_idt_y = ||net_g_y(real_x) - real_x|| * lambda_x * lambda_identity
                idt_y = net_g_y(real_x)
                loss_idt_y = idt_l1_loss_fn(idt_y, real_x) * lambda_x * lambda_identity

            else:
                loss_idt_y = 0

            # 2. real_x -> net_g_x -> fake_y -> net_d_x -> pred_real, real_label -> gan_loss_fn -> loss_g_x
            # loss_g_x = net_d(net_g_x(real_x)) - 1
            fake_y = net_g_x(real_x)
            pred_real = net_d_x(fake_y)
            loss_g_x = gan_loss_fn(pred_real, real_label.expand_as(pred_real))

            # 3. real_x -> net_g_x -> fake_y -> net_g_y -> rec_x, real_x -> loss_cycle_x
            # loss_cycle_x = ||net_g_y(net_g_x(real_x)) - real_x|| * lambda_x
            rec_x = net_g_y(fake_y)
            loss_cycle_x = cycle_l1_loss_fn(rec_x, real_x) * lambda_x

            return (fake_y, rec_x), (loss_idt_y, loss_g_x, loss_cycle_x)

        def update_d(real_x, fake_x, net_d_y, cacher):
            net_d_y.requires_grad_(True)

            # 4. real_x -> net_d_y -> pred_real, real_label -> loss_d_real
            # loss_d_real = net_d_y(real_x) - 1
            pred_real = net_d_y(real_x)
            loss_d_real = gan_loss_fn(pred_real, real_label.expand_as(pred_real))

            cacher.cache_batch(real_x)
            if random.uniform(0, 1) > 0.5:
                fake_x_old = cacher.get_batch(size=batch_size)
                fake_x_old = torch.stack(fake_x_old)
            else:
                fake_x_old = fake_x

            # 5. fake_x_old -> net_d_y -> pred_fake, fake_label -> loss_d_fake
            # loss_d_fake = net_d_y(fake_x_old) - 0
            pred_fake = net_d_y(fake_x_old.detach())
            loss_d_fake = gan_loss_fn(pred_fake, fake_label.expand_as(pred_fake))

            loss_d_x = (loss_d_real + loss_d_fake) * 0.5

            return loss_d_x

        j = 0
        for i in range(max_epoch):
            self.model.train()
            pbar = tqdm(dataloader, desc=f'train {i}/{max_epoch}')

            for rets in pbar:
                images_a = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_a = torch.stack(images_a)

                images_b = [torch.from_numpy(ret.pop('pix_image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_b = torch.stack(images_b)

                real_a = images_a.to(self.device)
                real_b = images_b.to(self.device)

                optimizer_g.zero_grad()
                (fake_b, rec_a), (loss_idt_b, loss_g_a, loss_cycle_a) = update_g(real_a, net_g_a, net_g_b, net_d_a, lambda_a)
                (fake_a, rec_b), (loss_idt_a, loss_g_b, loss_cycle_b) = update_g(real_b, net_g_b, net_g_a, net_d_b, lambda_b)
                loss_g = loss_g_a + loss_g_b + loss_cycle_a + loss_cycle_b + loss_idt_a + loss_idt_b
                loss_g.backward()
                optimizer_g.step()

                optimizer_d.zero_grad()
                loss_d_a = update_d(real_a, fake_a, net_d_b, fake_a_cacher)
                loss_d_b = update_d(real_b, fake_b, net_d_a, fake_b_cacher)
                loss_d = loss_d_a + loss_d_b
                loss_d.backward()
                optimizer_d.step()

                pbar.set_postfix({
                    'gen_iter': j,
                    'loss_g': f'{loss_g.item():.06}',
                    'loss_d': f'{loss_d.item():.06}',
                    # 'cpu_info': MemoryInfo.get_process_mem_info(),
                    # 'gpu_info': MemoryInfo.get_gpu_mem_info()
                })

                j += 1
                if j % save_period == 0:
                    vis_image = dict(
                        real_a=real_a,
                        fake_a=fake_a,
                        rec_a=rec_a,
                        real_b=real_b,
                        fake_b=fake_b,
                        rec_b=rec_b
                    )
                    ret = []
                    for name, images in vis_image.items():
                        images = images.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
                        ret.append([{'image': image, '_id': f'{j}_{name}.png'} for image in images])

                    ret = [r for r in zip(*ret)]
                    DataVisualizer(f'cache_data/{self.model_version}/{self.dataset_version}', verbose=True, stdout_method=self.logger.info)(*ret)
