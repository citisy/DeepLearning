import logging
import copy
import cv2
import torch
from torch import nn, optim
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from utils import os_lib, converter, configs, visualize
from utils.torch_utils import EarlyStopping, ModuleInfo, Export
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply, Apply, channel
from typing import List

MODEL = 1
WEIGHT = 2
ONNX = 3
JIT = 4
TRITON = 5


def setup_seed(seed=42):
    """42 is lucky number"""
    import torch.backends.cudnn as cudnn

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class BaseDataset(Dataset):
    def __init__(self, data, augment_func=None, complex_augment_func=None):
        self.data = data
        self.augment_func = augment_func
        self.complex_augment_func = complex_augment_func
        self.loader = os_lib.Loader(verbose=False)

    def __getitem__(self, idx):
        if self.complex_augment_func:
            return self.complex_augment_func(idx, self.data, self.process_one)
        return self.process_one(idx)

    def process_one(self, idx):
        ret = copy.deepcopy(self.data[idx])
        if isinstance(ret['image'], str):
            ret['image_path'] = ret['image']
            ret['image'] = self.loader.load_img(ret['image'])

        ret['ori_image'] = ret['image']
        ret['idx'] = idx

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        return list(batch)


class Process:
    dataset = BaseDataset
    setup_seed()

    def __init__(
            self,
            model=None, callable_model=None, optimizer=None, callable_optimizer=optim.Adam, model_params=None,
            input_size=None, out_features=None,
            model_version='', dataset_version='', device=0,
            log_dir=None, use_wandb=False, **kwargs
    ):
        self.model_version = model_version
        self.dataset_version = dataset_version
        self.model_dir = f'model_data/{self.model_version}'
        self.work_dir = f'{self.model_dir}/{self.dataset_version}'
        self.model_path = f'{self.work_dir}/weight.pth'
        self.save_result_dir = f'cache_data/{self.model_version}/{self.dataset_version}'
        os_lib.mk_dir(self.work_dir)

        configs.logger_init(log_dir)
        self.logger = logging.getLogger()

        if use_wandb:
            try:
                import wandb
            except ImportError:
                from utils.os_lib import FakeWandb
                wandb = FakeWandb()
                self.logger.warning('wandb import error, please check install')

        else:
            from utils.os_lib import FakeWandb
            wandb = FakeWandb()

        self.wandb = wandb
        self.wandb.init(
            project=self.model_version,
            name=self.dataset_version,
            dir=f'{self.work_dir}'
        )
        self.log_info = {}

        self.logger.info(f'{self.model_version = }')
        self.logger.info(f'{self.dataset_version = }')

        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu") if device is not None else 'cpu'
        self.model = model or callable_model()
        self.callable_model = callable_model
        model_params = model_params or self.model.parameters()
        self.optimizer = optimizer or callable_optimizer(model_params)
        self.callable_optimizer = callable_optimizer
        self.stopper = EarlyStopping(patience=10, min_epoch=10, stdout_method=self.logger.info)

        self.input_size = input_size or self.model.input_size
        self.out_features = out_features

        self.__dict__.update(kwargs)

    def model_info(self, depth=None, human_readable=True):
        profile = ModuleInfo.profile_per_layer(self.model, depth=depth)
        s = f'module info: \n{"name":<20}{"module":<40}{"params":>10}{"grads":>10}  {"args"}\n'

        for p in profile:
            params = p[2]["params"]
            grads = p[2]["grads"]
            args = p[2]["args"]
            if human_readable:
                params = visualize.TextVisualize.num_to_human_readable_str(params)
                grads = visualize.TextVisualize.num_to_human_readable_str(grads)
                args = visualize.TextVisualize.dict_to_str(args)
                args = args or '-'
            s += f'{p[0]:<20}{p[1]:<40}{params:>10}{grads:>10}  {args}\n'

        params = sum([p[2]["params"] for p in profile])
        grads = sum([p[2]["grads"] for p in profile])
        if human_readable:
            params = visualize.TextVisualize.num_to_human_readable_str(params)
            grads = visualize.TextVisualize.num_to_human_readable_str(grads)

        s += f'{"sum":<20}{"":<40}{params:>10}{grads:>10}\n'

        self.logger.info(s)

    def run(self, max_epoch=100, train_batch_size=16, predict_batch_size=None, save_period=None, fit_kwargs=dict(), metric_kwargs=dict()):
        self.model_info()
        fit_kwargs.setdefault('metric_kwargs', metric_kwargs)

        self.fit(
            max_epoch, train_batch_size, save_period,
            num_workers=min(train_batch_size, 16),
            **fit_kwargs
        )
        self.save(self.model_path)

        # self.load(self.model_path)
        # self.load(f'{self.model_dir}/{self.dataset_version}_last.pth')

        r = self.metric(
            batch_size=predict_batch_size or train_batch_size,
            num_workers=min(predict_batch_size or train_batch_size, 16),
            **metric_kwargs
        )
        for k, v in r.items():
            self.logger.info({k: v})

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def metric(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def single_predict(self, image: np.ndarray, **kwargs):
        raise NotImplementedError

    def batch_predict(self, images: List[np.ndarray], batch_size=16, **kwargs):
        raise NotImplementedError

    def fragment_predict(self, image: np.ndarray, **kwargs):
        """predict large image. Tear picture to pieces to predict, and then merge the results"""
        raise NotImplementedError

    def on_train_start(self, batch_size, metric_kwargs=dict(), **dataloader_kwargs):
        metric_kwargs = configs.merge_dict(dataloader_kwargs, metric_kwargs)
        metric_kwargs.setdefault('batch_size', batch_size)

        train_data = self.get_train_data()

        train_dataset = self.dataset(
            train_data,
            augment_func=self.data_augment,
            complex_augment_func=self.complex_data_augment if hasattr(self, 'complex_data_augment') else None
        )

        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            # sampler=sampler,
            pin_memory=True,
            batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            **dataloader_kwargs
        )

        val_data = self.get_val_data()
        val_dataset = self.dataset(val_data, augment_func=self.val_data_augment)

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=val_dataset.collate_fn,
            **dataloader_kwargs
        )

        self.model.to(self.device)

        return train_dataloader, val_dataloader, metric_kwargs

    def on_train_step_end(self, *args, **kwargs):
        raise NotImplementedError

    def on_train_epoch_end(self, epoch, save_period, mean_loss, val_dataloader, **metric_kwargs):
        self.logger.info(f'train log: epoch: {epoch}, mean_loss: {mean_loss}')
        self.log_info = {'epoch': epoch, 'mean_loss': mean_loss}

        if save_period and epoch % save_period == save_period - 1:
            self.save(f'{self.model_dir}/{self.dataset_version}/last.pth')

            result = self.metric(val_dataloader, cur_epoch=epoch, **metric_kwargs)
            score = result['score']
            self.log_info['score'] = score
            self.logger.info(f"val log: epoch: {epoch}, score: {score}")

            if score > self.stopper.best_fitness:
                self.save(f'{self.model_dir}/{self.dataset_version}/best.pth')

            if self.stopper(epoch=epoch, fitness=score):
                return True

        self.wandb.log(self.log_info)

    def on_train_end(self, *args, **kwargs):
        raise NotImplementedError

    def on_val_start(self, batch_size, **dataloader_kwargs):
        val_data = self.get_val_data()
        val_dataset = self.dataset(val_data, augment_func=self.val_data_augment)

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=val_dataset.collate_fn,
            **dataloader_kwargs
        )

        return val_dataloader

    def on_val_step_end(self, *args, **kwargs):
        raise NotImplementedError

    def get_train_data(self, *args, **kwargs):
        raise NotImplementedError

    def get_val_data(self, *args, **kwargs):
        raise NotImplementedError

    def data_augment(self, ret) -> dict:
        raise NotImplementedError

    def val_data_augment(self, ret) -> dict:
        raise NotImplementedError

    def val_data_restore(self, ret) -> dict:
        raise NotImplementedError

    def save(self, save_path, save_type=WEIGHT, verbose=True, **kwargs):
        os_lib.mk_dir(Path(save_path).parent)

        if save_type == MODEL:
            torch.save(self.model, save_path)
        elif save_type == WEIGHT:
            ckpt = dict(
                model=self.model.state_dict(),
                **kwargs
            )
            torch.save(ckpt, save_path)
        elif save_type == JIT:
            trace_input = torch.rand(1, 3, self.input_size, self.input_size).to(self.device)
            model = Export.to_jit(self.model, trace_input)
            model.save(save_path)
        elif save_type == TRITON:
            pass
        else:
            raise ValueError(f'dont support {save_type = }')

        if verbose:
            self.logger.info(f'Successfully saved to {save_path} !')

    def load(self, save_path, save_type=WEIGHT, verbose=True):
        if save_type == MODEL:
            self.model = torch.load(save_path, map_location=self.device)
        elif save_type == WEIGHT:
            ckpt = torch.load(save_path, map_location=self.device)
            if self.model is None:
                self.model = self.callable_model(**ckpt['config'])
            self.model.load_state_dict(ckpt['model'], strict=False)
        elif save_type == JIT:
            self.model = torch.jit.load(save_path, map_location=self.device)
        else:
            raise ValueError(f'dont support {save_type = }')

        if verbose:
            self.logger.info(f'Successfully load {save_path} !')


class ParamsSearch:
    """
    Usage:
        .. code-block:: python

            from examples.image_classifier import ClsProcess, ImageNet
            from models.image_classifier.ResNet import Model
            from torch import optim

            class ResNet_ImageNet(ClsProcess, ImageNet):
                pass

            params_search = ParamsSearch(
                process=ResNet_ImageNet,
                var_instance=dict(
                    model=Model,
                    callable_optimizer=lambda **kwargs: lambda params: optim.Adam(params, **kwargs),
                ),
                var_params=dict(
                    model=dict(input_size=(224, 256)),
                    callable_optimizer=dict(lr=(0.001, 0.01))
                ),
                const_params=dict(
                    model=dict(in_ch=3, out_features=2)
                ),
                run_kwargs=dict(max_epoch=100, save_period=4),
                process_kwargs=dict(use_wandb=True),
                model_version='ResNet',
                dataset_version='ImageNet2012.ps',
            )
            params_search.run()
    """

    def __init__(
            self,
            process, keys, var_params, const_params=dict(),
            process_kwargs=dict(), run_kwargs=dict(),
            model_version='', dataset_version='',
    ):
        self.process = process
        self.var_instance = keys
        self.var_params = var_params
        self.const_params = const_params
        self.process_kwargs = process_kwargs
        self.run_kwargs = run_kwargs
        self.model_version = model_version
        self.dataset_version = dataset_version

        self.var_params = {k: configs.permute_obj(var_p) for k, var_p in self.var_params.items()}
        keys = []
        self.total_params = [[]]
        for k, var_ps in self.var_params.items():
            const_p = self.const_params.get(k, {})
            keys.append(k)
            self.total_params = [_ + [(var_p, const_p)] for _ in self.total_params for var_p in var_ps]

        self.keys = keys

    def run(self):
        kwargs = copy.deepcopy(self.process_kwargs)
        for _ in self.total_params:
            sub_version = ''
            for key, (var_p, const_p) in zip(self.keys, _):
                for k, v in var_p.items():
                    sub_version += f'{k}={v},'

                params = configs.merge_dict(var_p, const_p)
                if key in self.var_instance:
                    ins = self.var_instance[key]
                    kwargs[key] = ins(**params)
                else:
                    kwargs.update(params)

            sub_version = sub_version[:-1]

            kwargs['model_version'] = self.model_version
            kwargs['dataset_version'] = f'{self.dataset_version}/{sub_version}'
            kwargs['log_dir'] = f'model_data/{self.model_version}/{self.dataset_version}/{sub_version}/logs'
            process = self.process(**kwargs)
            process.run(**self.run_kwargs)
