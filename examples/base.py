import logging
import copy
import time
import cv2
import torch
from torch import nn, optim
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from utils import os_lib, converter, configs, visualize, log_utils
from utils.torch_utils import EarlyStopping, ModuleInfo, Export
from data_parse.cv_data_parse.data_augmentation import crop, scale, geometry, pixel_perturbation, RandomApply, Apply, channel
from typing import List
from datetime import datetime

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
        else:
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


class IterDataset(BaseDataset):
    length = None

    def process_one(self, *args):
        ret = next(self.data)
        if isinstance(ret['image'], str):
            ret['image_path'] = ret['image']
            ret['image'] = self.loader.load_img(ret['image'])

        ret['ori_image'] = ret['image']

        if self.augment_func:
            ret = self.augment_func(ret)

        return ret

    def __len__(self):
        return self.length


class MixDataset(Dataset):
    def __init__(self, obj, **kwargs):
        self.datasets = []
        for data, dataset_instance in obj:
            self.datasets.append(dataset_instance(data, **kwargs))

        self.nums = [len(_) for _ in self.datasets]

    def __getitem__(self, idx):
        for n, dataset in zip(self.nums, self.datasets):
            idx -= n
            if idx < 0:
                return dataset[idx]

    def __len__(self):
        return sum(self.nums)

    @staticmethod
    def collate_fn(batch):
        return list(batch)


class Process:
    train_dataset_ins = BaseDataset
    val_dataset_ins = BaseDataset
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

        log_utils.logger_init(log_dir)
        self.logger = log_utils.get_logger()

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
        self.log_info = {}

        self.logger.info(f'{self.model_version = }')
        self.logger.info(f'{self.dataset_version = }')

        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu") if device is not None else 'cpu'
        # note that, it must be set device before load_state_dict()
        self.model = (model or callable_model()).to(self.device)
        self.callable_model = callable_model
        model_params = model_params or self.model.parameters()
        self.optimizer = optimizer or callable_optimizer(model_params)
        self.callable_optimizer = callable_optimizer
        self.stopper = EarlyStopping(patience=10, min_epoch=10, ignore_min_score=0.1, stdout_method=self.logger.info)

        self.input_size = input_size or self.model.input_size
        self.out_features = out_features
        self.start_epoch = 0
        self.date = datetime.now().isoformat()

        self.__dict__.update(kwargs)

    def model_info(self, **kwargs):
        return self._model_info(self.model, **kwargs)

    def _model_info(self, model, depth=None, human_readable=True):
        profile = ModuleInfo.profile_per_layer(model, depth=depth)
        cols = ('name', 'module', 'params', 'grads', 'args')
        lens = [-1] * len(cols)
        infos = []
        for p in profile:
            info = (
                p[0],
                p[1],
                visualize.TextVisualize.num_to_human_readable_str(p[2]["params"]) if human_readable else p[2]["params"],
                visualize.TextVisualize.num_to_human_readable_str(p[2]["grads"]) if human_readable else p[2]["grads"],
                visualize.TextVisualize.dict_to_str(p[2]["args"])
            )
            infos.append(info)
            for i, s in enumerate(info):
                l = len(str(s))
                if lens[i] < l:
                    lens[i] = l

        template = ''
        for l in lens:
            template += f'%-{l + 3}s'

        s = 'module info: \n'
        s += template % cols + '\n'
        s += template % tuple('-' * l for l in lens) + '\n'

        for info in infos:
            s += template % info + '\n'

        params = sum([p[2]["params"] for p in profile])
        grads = sum([p[2]["grads"] for p in profile])
        if human_readable:
            params = visualize.TextVisualize.num_to_human_readable_str(params)
            grads = visualize.TextVisualize.num_to_human_readable_str(grads)

        s += template % tuple('-' * l for l in lens) + '\n'
        s += template % ('sum', '', params, grads, '')
        self.logger.info(s)
        return infos

    def run(self, max_epoch=100, train_batch_size=16, predict_batch_size=None, save_period=None, fit_kwargs=dict(), metric_kwargs=dict()):
        self.model_info()
        fit_kwargs.setdefault('metric_kwargs', metric_kwargs)

        self.fit(
            max_epoch,
            train_batch_size,
            save_period=save_period,
            num_workers=min(train_batch_size, 16),
            **fit_kwargs
        )

        ckpt = {
            'start_epoch': max_epoch + 1,
            'wandb_id': self.wandb_run.id,
            'date': datetime.now().isoformat()
        }

        self.save(self.model_path, save_type=WEIGHT, **ckpt)

        # self.load(self.model_path, save_type=WEIGHT)
        # self.load(f'{self.model_dir}/{self.dataset_version}/last.pth', save_type=WEIGHT)

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

    def on_train_start(self, batch_size, metric_kwargs=dict(), return_val_dataloader=True, **dataloader_kwargs):
        metric_kwargs = configs.merge_dict(dataloader_kwargs, metric_kwargs)
        metric_kwargs.setdefault('batch_size', batch_size)

        self.wandb_run = self.wandb.init(
            project=self.model_version,
            name=self.dataset_version,
            dir=f'{self.work_dir}',
            id=self.wandb_id if hasattr(self, 'wandb_id') else None,
            reinit=True
        )
        self.wandb_id = self.wandb_run.id

        train_data = self.get_train_data()

        train_dataset = self.train_dataset_ins(
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

        if return_val_dataloader:
            val_data = self.get_val_data()
            val_dataset = self.val_dataset_ins(val_data, augment_func=self.val_data_augment)

            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                collate_fn=val_dataset.collate_fn,
                **dataloader_kwargs
            )
        else:
            val_dataloader = None

        self.model.to(self.device)

        return train_dataloader, val_dataloader, metric_kwargs

    def on_train_step_end(self, *args, **kwargs):
        raise NotImplementedError

    def on_train_epoch_end(self, epoch, save_period=None, val_dataloader=None,
                           losses=None, epoch_start_time=None, **metric_kwargs):
        end_flag = False
        self.log_info = {'epoch': epoch}

        if losses is not None:
            for k, v in losses.items():
                self.log_info[f'loss/{k}'] = v
                if np.isnan(v) or np.isinf(v):
                    end_flag = True
                    self.logger.info(f'Train will be stop soon, got {v} value from {k}')

        if epoch_start_time is not None:
            self.log_info['time_consume'] = (time.time() - epoch_start_time) / 60

        self.logger.info(f'train log: {self.log_info}')
        ckpt = {
            'start_epoch': epoch + 1,
            'wandb_id': self.wandb_id,
            'date': datetime.now().isoformat()
        }

        self.save(f'{self.model_dir}/{self.dataset_version}/last.pth', save_type=WEIGHT, only_model=False, **ckpt)

        if save_period and epoch % save_period == save_period - 1:
            result = self.metric(val_dataloader, cur_epoch=epoch, **metric_kwargs)
            score = result['score']
            self.log_info['val_score'] = score
            self.logger.info(f"val log: epoch: {epoch}, score: {score}")

            if score > self.stopper.best_score:
                self.save(f'{self.model_dir}/{self.dataset_version}/best.pth', save_type=WEIGHT, only_model=False, **ckpt)

            end_flag = end_flag or self.stopper(epoch=epoch, score=score)

        self.wandb.log(self.log_info)
        return end_flag

    def on_train_end(self, *args, **kwargs):
        raise NotImplementedError

    def on_val_start(self, batch_size, **dataloader_kwargs):
        val_data = self.get_val_data()
        val_dataset = self.train_dataset_ins(val_data, augment_func=self.val_data_augment)

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

    def save(self, save_path, save_type=WEIGHT, verbose=True, only_model=True, **kwargs):
        os_lib.mk_dir(Path(save_path).parent)

        if save_type == MODEL:
            if only_model:
                torch.save(self.model, save_path)
            else:
                torch.save(self.model, save_path['model'])
                torch.save(self.optimizer, save_path['optimizer'])

        elif save_type == WEIGHT:
            ckpt = dict(
                model=self.model.state_dict(),
            )
            if not only_model:
                ckpt.update(
                    optimizer=self.optimizer.state_dict(),
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

            if 'model' in ckpt:
                if self.model is None:
                    self.model = self.callable_model(**ckpt['model_config'])
                self.model.load_state_dict(ckpt.pop('model'), strict=False)

            if 'optimizer' in ckpt:
                if self.optimizer is None:
                    self.optimizer = self.callable_optimizer(**ckpt['optimizer_config'])
                self.optimizer.load_state_dict(ckpt.pop('optimizer'))

            self.__dict__.update(ckpt)

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

            ######## example 1 ########
            from examples.image_classifier import ClsProcess, ImageNet
            from models.image_classifier.ResNet import Model
            from torch import optim

            class ResNet_ImageNet(ClsProcess, ImageNet):
                '''define your own process'''

            params_search = ParamsSearch(
                process=ResNet_ImageNet,
                params=dict(
                    model=dict(
                        instance=Model,
                        const=dict(in_ch=3, out_features=2),
                        var=dict(input_size=(224, 256, 320))
                    ),
                    callable_optimizer=dict(
                        instance=lambda **kwargs: lambda params: optim.Adam(params, **kwargs),
                        var=dict(lr=(0.001, 0.01))
                    ),
                    sys=dict(
                        var=dict(lf=[
                            lambda x, max_epoch, lrf: (1 - x / max_epoch) * (1.0 - lrf) + lrf,
                            lambda x, max_epoch, lrf: ((1 - math.cos(x * math.pi / max_epoch)) / 2) * (lrf - 1) + 1,
                        ])
                    )
                ),
                run_kwargs=dict(max_epoch=100, save_period=4),
                process_kwargs=dict(use_wandb=True),
                model_version='ResNet',
                dataset_version='ImageNet2012.ps',
            )
            # there is 3*2*2 test group
            params_search.run()

            ######## example 2 ########
            from models.object_detection.YoloV5 import Model, head_config, make_config, default_model_multiple
            params_search = ParamsSearch(
                process=Process,
                params=dict(
                    model=dict(
                        instance=Model,
                        const=dict(
                            n_classes=20,
                            in_module_config=dict(in_ch=3, input_size=640),
                            head_config=head_config
                        ),
                        var=[
                            {k: [v] for k, v in make_config(**default_model_multiple['yolov5n']).items()},
                            {k: [v] for k, v in make_config(**default_model_multiple['yolov5s']).items()},
                            {'head_config.anchor_t': [3, 4, 5]}
                        ]
                    ),
                    sys=dict(
                        const=dict(
                            input_size=None,
                            device=0,
                            cls_alias=classes
                        ),
                    )
                ),
                run_kwargs=dict(max_epoch=100, save_period=4, metric_kwargs=dict(visualize=True, max_vis_num=8)),
                process_kwargs=dict(use_wandb=True),
                model_version='yolov5-test',
                dataset_version='Voc.ps',
            )
            # there are 5 test groups
            params_search.run()

    """

    def __init__(
            self,
            process, params=dict(),
            process_kwargs=dict(), run_kwargs=dict(),
            model_version='', dataset_version='',
    ):
        self.process = process
        self.process_kwargs = process_kwargs
        self.run_kwargs = run_kwargs
        self.model_version = model_version
        self.dataset_version = dataset_version

        var_params = {k: v['var'] for k, v in params.items() if 'var' in v}
        self.var_params = {k: configs.permute_obj(var_p) for k, var_p in var_params.items()}
        self.const_params = {k: v['const'] for k, v in params.items() if 'const' in v}
        self.var_instance = {k: v['instance'] for k, v in params.items() if 'instance' in v}
        self.total_params = [[]]
        keys = set(self.var_params.keys()) | set(self.const_params.keys())
        for k in keys:
            var_ps = self.var_params.get(k, [{}])
            const_p = self.const_params.get(k, {})
            self.total_params = [_ + [(var_p, const_p)] for _ in self.total_params for var_p in var_ps]

        self.keys = keys

    def run(self):
        kwargs = copy.deepcopy(self.process_kwargs)
        for _ in self.total_params:
            sub_version = ''
            info_msg = ''
            for key, (var_p, const_p) in zip(self.keys, _):
                tmp_var_p = configs.collapse_dict(var_p)
                for k, v in tmp_var_p.items():
                    if len(str(v)) > 8:
                        s = converter.DataConvert.str_to_md5(str(v))
                        sub_version += f'{k}={s[:6]};'
                    else:
                        sub_version += f'{k}={v};'
                    info_msg += f'{k}={v};'

                var_p = configs.expand_dict(var_p)
                params = configs.merge_dict(var_p, const_p)
                if key in self.var_instance:
                    ins = self.var_instance[key]
                    kwargs[key] = ins(**params)
                else:
                    kwargs.update(params)

            sub_version = sub_version[:-1]
            info_msg = info_msg[:-1]

            kwargs['model_version'] = self.model_version
            kwargs['dataset_version'] = f'{self.dataset_version}/{sub_version}'
            kwargs['log_dir'] = f'model_data/{self.model_version}/{self.dataset_version}/{sub_version}/logs'
            process = self.process(**kwargs)
            process.logger.info(info_msg)
            process.run(**self.run_kwargs)
