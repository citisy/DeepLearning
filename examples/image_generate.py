from .base import Process, BaseDataset
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.torch_utils import EarlyStopping, ModuleInfo, Export
from data_parse.cv_data_parse.base import DataRegister, DataVisualizer
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply, Apply, channel


class IgDataset(BaseDataset):
    pass


class IgProcess(Process):
    dataset = BaseDataset

    def model_info(self, depth=None):
        profile = ModuleInfo.profile_per_layer(self.model.net_d, depth=depth)
        s = f'net d module info: \n{"name":<20}{"module":<40}{"params":>10}{"grads":>10}\n'

        for p in profile:
            s += f'{p[0]:<20}{p[1]:<40}{p[2]["params"]:>10}{p[2]["grads"]:>10}\n'

        self.logger.info(s)

        profile = ModuleInfo.profile_per_layer(self.model.net_g, depth=depth)
        s = f'net g module info: \n{"name":<20}{"module":<40}{"params":>10}{"grads":>10}\n'

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

                # see also https://github.com/martinarjovsky/WassersteinGAN/issues/18
                for p in net_d.parameters():
                    p.data.clamp_(-0.01, 0.01)

                net_d.zero_grad()

                # 1. real_x -> net_d -> pred_real -> loss_d_real -> gradient_descent
                real_x = images
                pred_real = net_d(real_x)

                # note that, wgan backward without loss function,
                # see also https://github.com/martinarjovsky/WassersteinGAN/issues/9
                loss_d_real = pred_real
                loss_d_real.backward()

                # 2. noise -> net_g -> fake_x -> net_d -> pred_fake -> loss_d_fake -> gradient_ascent
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
                    # update G
                    net_d.requires_grad_(False)
                    net_g.zero_grad()

                    # noise -> net_g -> fake_x -> net_d -> pred_fake -> loss_g -> gradient_descent
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


class Pix2pix_facade(IgProcess):
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
        ret.update(dst=self.input_size)
        ret.update(Apply([
            scale.LetterBox(),
            pixel_perturbation.MinMax(),
            # pixel_perturbation.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            channel.HWC2CHW()
        ])(**ret))

        pix_image = ret['pix_image']
        pix_image = scale.Proportion().apply(pix_image, ret['scale.Proportion']['p'])['image']
        pix_image = crop.Pad().apply(pix_image, ret['crop.Pad'])['image']
        h, w = pix_image.shape[:-1]
        c = ret['crop.Crop']
        l, r, t, d = c['l'], c['r'], c['t'], c['d']
        pix_image = pix_image[t:h - d, l:w - r]
        pix_image = channel.HWC2CHW()(pix_image)['image']

        ret['pix_image'] = pix_image

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
        optimizer_g = torch.optim.Adam(net_g.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(net_d.parameters(), lr=0.0002, betas=(0.5, 0.999))

        gan_loss_fn = nn.MSELoss()
        real_label = torch.tensor(1., device=self.device)
        fake_label = torch.tensor(0., device=self.device)
        l1_loss_fn = torch.nn.L1Loss()
        lambda_l1 = 100.0

        j = 0
        for i in range(max_epoch):
            net_g.train()
            net_d.train()
            pbar = tqdm(dataloader, desc=f'train {i}/{max_epoch}')

            for rets in pbar:
                images_a = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_a = torch.stack(images_a)

                images_b = [torch.from_numpy(ret.pop('pix_image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in rets]
                images_b = torch.stack(images_b)

                real_a = images_a.to(self.device)
                real_b = images_b.to(self.device)

                # 1. real_a -> net_g -> fake_b -> loss_g_l1 -> gradient_descent
                fake_b = net_g(real_a)

                # update D
                net_d.requires_grad_(True)
                optimizer_d.zero_grad()

                # 2. real_a + fake_b -> net_d -> pred_fake -> loss_d_fake -> fake_label
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = net_d(fake_ab.detach())
                loss_d_fake = gan_loss_fn(pred_fake, fake_label.expand_as(pred_fake))

                # 3. real_a + real_b -> net_d -> pred_real -> loss_d_real -> real_label
                real_ab = torch.cat((real_a, real_b), 1)
                pred_real = net_d(real_ab)
                loss_d_real = gan_loss_fn(pred_real, real_label.expand_as(pred_real))

                loss_d = (loss_d_fake + loss_d_real) * 0.5
                loss_d.backward()
                optimizer_d.step()

                # update G
                net_d.requires_grad_(False)
                optimizer_g.zero_grad()

                #  real_a + fake_b -> net_d -> pred_fake -> loss_g_gan -> real_label
                pred_fake = net_d(fake_ab)
                loss_g_gan = gan_loss_fn(pred_fake, real_label.expand_as(pred_fake))
                loss_g_l1 = l1_loss_fn(fake_b, real_b) * lambda_l1

                loss_g = loss_g_gan + loss_g_l1
                loss_g.backward()
                optimizer_g.step()

                pbar.set_postfix({
                    'gen_iter': j,
                    'G_GAN': f'{loss_g_gan.item():.06}',
                    'G_L1': f'{loss_g_l1.item():.06}',
                    'D_real': f'{loss_d_real.item():06}',
                    'D_fake': f'{loss_d_fake.item():06}'
                    # 'cpu_info': MemoryInfo.get_process_mem_info(),
                    # 'gpu_info': MemoryInfo.get_gpu_mem_info()
                })

                j += 1
                if j % save_period == 0:
                    real_a = real_a.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
                    fake_b = fake_b.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
                    real_b = real_b.permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()
                    ret = [[
                        {'image': ra, '_id': f'{j}_real_A.png'},
                        {'image': fb, '_id': f'{j}_fake_B.png'},
                        {'image': rb, '_id': f'{j}_real_B.png'}
                    ] for ra, fb, rb in zip(real_a, fake_b, real_b)]

                    DataVisualizer(f'cache_data/{self.dataset_version}', verbose=True, stdout_method=self.logger.info)(*ret)
