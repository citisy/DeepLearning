import itertools
import torch
from torch import optim
from data_parse.cv_data_parse.data_augmentation import scale, channel, Apply, pixel_perturbation
from data_parse import DataRegister
from processor import DataHooks
from .image_generation import GanProcess, GanOptimizer
from utils import os_lib


class Facade(DataHooks):
    dataset_version = 'facade'
    data_dir = 'data/cmp_facade'

    train_data_num = None
    val_data_num = None

    input_size = 256
    in_ch = 3

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.cv_data_parse.datasets.cmp_facade import Loader
        loader = Loader(self.data_dir)

        if train:
            return loader(
                set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY, generator=False,
                max_size=self.train_data_num
            )[0]

        else:
            return loader(
                set_type=DataRegister.TEST, image_type=DataRegister.ARRAY, generator=False,
                max_size=self.val_data_num,
            )[0]

    aug = Apply([
        scale.LetterBox(),
        pixel_perturbation.MinMax(),
        # pixel_perturbation.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        channel.HWC2CHW()
    ])

    def data_augment(self, ret, train=True) -> dict:
        ret.update(dst=self.input_size)
        ret.update(self.aug(**ret))
        label_mask = ret['label_mask']
        label_mask = self.aug.apply_image(label_mask, ret)
        ret['label_mask'] = label_mask

        return ret

    def val_data_restore(self, ret) -> dict:
        ret = self.aug.restore(ret)
        return ret


class Pix2pix(GanProcess):
    model_version = 'Pix2pix'

    def set_model(self):
        from models.image_translation.pix2pix import Model

        self.model = Model(
            in_ch=self.in_ch,
            input_size=self.input_size
        )

    def set_optimizer(self, lr_g=0.00002, betas_g=(0.5, 0.999), lr_d=0.00002, betas_d=(0.5, 0.999), **kwargs):
        optimizer_g = optim.Adam(self.model.net_g.parameters(), lr=lr_g, betas=betas_g)
        optimizer_d = optim.Adam(self.model.net_d.parameters(), lr=lr_d, betas=betas_d)
        self.optimizer = GanOptimizer(optimizer_d, optimizer_g)

    def get_model_inputs(self, loop_inputs, train=True):
        images_a = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
        images_a = torch.stack(images_a)

        images_b = [torch.from_numpy(ret.pop('label_mask')).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
        images_b = torch.stack(images_b)

        return dict(
            real_a=images_a,
            real_b=images_b
        )

    def on_train_step(self, loop_objs, **kwargs) -> dict:
        loop_objs = self.get_model_inputs(loop_objs)
        inputs = self.get_model_inputs(loop_objs)
        real_a = inputs['real_a']
        real_b = inputs['real_b']
        fake_b = self.model.net_g(real_a)
        fake_ab = torch.cat((real_a, fake_b), 1)

        self.optimizer.optimizer_d.zero_grad()
        loss_d = self.model.loss_d(real_a, real_b, fake_ab)
        loss_d.backward()
        self.optimizer.optimizer_d.step()

        self.optimizer.optimizer_g.zero_grad()
        loss_g = self.model.loss_g(real_b, fake_b, fake_ab)
        loss_g.backward()
        self.optimizer.optimizer_g.step()

        return {
            'loss.g': loss_g,
            'loss.d': loss_d,
        }

    def on_val_start(self, val_dataloader=None, dataloader_kwargs=dict(), **kwargs):
        loop_objs, process_kwargs = super().on_val_start(**kwargs)
        if val_dataloader is None:
            val_dataloader = self.get_val_dataloader(**dataloader_kwargs)
        process_kwargs['val_dataloader'] = val_dataloader
        return loop_objs, process_kwargs

    def on_val_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs)
        real_a = inputs['real_a']
        real_b = inputs['real_b']

        model_results = {}
        for name, model in self.models.items():
            fake_b = model.net_g(real_a)

            r = dict(
                fake_b=fake_b
            )

            if name == self.model_name:
                r.update(
                    real_a=real_a,
                    real_b=real_b,
                )

            for k, v in r.items():
                r[k] = v.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

            model_results[name] = r

        return model_results


class Pix2pix_facade(Pix2pix, Facade):
    """
    Usage:
        .. code-block:: python

            from bundles.image_translate import Pix2pix_facade as Process

            Process().run(max_epoch=1000, train_batch_size=64, check_period=2000)
    """


class CycleGan(GanProcess):
    model_version = 'CycleGan'

    def set_model(self):
        from models.image_translation.CycleGan import Model
        self.model = Model(
            in_ch=self.in_ch,
            input_size=self.input_size
        )

    def set_optimizer(self, lr_g=0.00005, betas_g=(0.5, 0.999), lr_d=0.00005, betas_d=(0.5, 0.999), **kwargs):
        optimizer_g = optim.Adam(itertools.chain(self.model.net_g_a.parameters(), self.model.net_g_b.parameters()), lr=lr_g, betas=betas_g)
        optimizer_d = optim.Adam(itertools.chain(self.model.net_d_a.parameters(), self.model.net_d_b.parameters()), lr=lr_d, betas=betas_d)
        self.optimizer = GanOptimizer(optimizer_d, optimizer_g)

    def model_info(self, depth=None, **kwargs):
        from utils.torch_utils import ModuleInfo

        modules = dict(
            d_a=self.model.net_d_a,
            d_b=self.model.net_d_b,
            g_a=self.model.net_g_a,
            g_b=self.model.net_g_b
        )

        for key, module in modules.items():
            s, infos = ModuleInfo.std_profile(module, **kwargs)
            self.log(f'net {key} module info:')
            self.log(s)

    def on_train_start(self, **kwargs):
        loop_objs, process_kwargs = super().on_train_start(**kwargs)
        loop_objs['fake_a_cacher'] = os_lib.MemoryCacher(max_size=50, verbose=False)
        loop_objs['fake_b_cacher'] = os_lib.MemoryCacher(max_size=50, verbose=False)
        return loop_objs, process_kwargs

    lambda_a = 10
    lambda_b = 10

    def get_model_inputs(self, loop_inputs, train=True):
        images_a = [torch.from_numpy(ret.pop('image')).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
        images_a = torch.stack(images_a)

        images_b = [torch.from_numpy(ret.pop('label_mask')).to(self.device, non_blocking=True, dtype=torch.float) for ret in loop_inputs]
        images_b = torch.stack(images_b)

        return dict(
            real_a=images_a,
            real_b=images_b
        )

    def on_train_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        optimizer_d, optimizer_g = self.optimizer.optimizer_d, self.optimizer.optimizer_g

        net_g_a = self.model.net_g_a
        net_g_b = self.model.net_g_b
        net_d_a = self.model.net_d_a
        net_d_b = self.model.net_d_b

        inputs = self.get_model_inputs(loop_inputs)
        real_a = inputs['real_a']
        real_b = inputs['real_b']

        optimizer_g.zero_grad()
        (fake_b, rec_a), (loss_idt_b, loss_g_a, loss_cycle_a) = self.model.loss_g(real_a, net_g_a, net_g_b, net_d_a, self.lambda_a)
        (fake_a, rec_b), (loss_idt_a, loss_g_b, loss_cycle_b) = self.model.loss_g(real_b, net_g_b, net_g_a, net_d_b, self.lambda_b)
        loss_g = loss_g_a + loss_g_b + loss_cycle_a + loss_cycle_b + loss_idt_a + loss_idt_b
        loss_g.backward()
        optimizer_g.step()

        optimizer_d.zero_grad()
        loss_d_a = self.model.loss_d(real_a, fake_a, net_d_b, loop_objs['fake_a_cacher'])
        loss_d_b = self.model.loss_d(real_b, fake_b, net_d_a, loop_objs['fake_b_cacher'])
        loss_d = loss_d_a + loss_d_b
        loss_d.backward()
        optimizer_d.step()

        return {
            'loss.g': loss_g,
            'loss.d': loss_d,
        }

    def on_val_start(self, val_dataloader=None, dataloader_kwargs=dict(), **kwargs):
        loop_objs, process_kwargs = super().on_val_start(**kwargs)
        if val_dataloader is None:
            val_dataloader = self.get_val_dataloader(**dataloader_kwargs)
        process_kwargs['val_dataloader'] = val_dataloader
        return loop_objs, process_kwargs

    def on_val_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs)
        real_a = inputs['real_a']
        real_b = inputs['real_b']

        model_results = {}
        for name, model in self.models.items():
            fake_b = model.net_g_a(real_a)
            fake_a = model.net_g_b(real_b)

            rec_a = model.net_g_b(fake_b)
            rec_b = model.net_g_a(fake_a)

            r = dict(
                fake_a=fake_a,
                rec_a=rec_a,
                fake_b=fake_b,
                rec_b=rec_b
            )

            if name == self.model_name:
                r.update(
                    real_a=real_a,
                    real_b=real_b,
                )

            for k, v in r.items():
                r[k] = v.data.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to("cpu", torch.uint8).numpy()

            model_results[name] = r

        return model_results


class CycleGan_facade(CycleGan, Facade):
    """
    Usage:
        .. code-block:: python

            from bundles.image_translate import CycleGan_facade as Process

            Process().run(max_epoch=1000, train_batch_size=8, check_period=500)
    """
