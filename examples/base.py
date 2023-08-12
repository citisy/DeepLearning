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

configs.logger_init()

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

    def __init__(self, model=None, optimizer=None, model_version='', dataset_version='', device=0, input_size=224, out_features=None):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu") if device is not None else 'cpu'
        self.model = model
        self.optimizer = optimizer or optim.Adam(self.model.parameters())
        self.model_version = model_version
        self.dataset_version = dataset_version
        self.model_dir = f'model_data/{self.model_version}'
        os_lib.mk_dir(self.model_dir)
        self.model_path = f'{self.model_dir}/{self.dataset_version}/weight.pth'
        self.save_result_dir = f'cache_data/{self.dataset_version}'
        self.input_size = input_size
        self.out_features = out_features
        self.logger = logging.getLogger()

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

    def run(self, max_epoch=100, train_batch_size=16, predict_batch_size=None, save_period=None):
        self.model_info()
        data = self.get_train_data()

        dataset = self.dataset(
            data,
            augment_func=self.data_augment,
            complex_augment_func=self.complex_data_augment if hasattr(self, 'complex_data_augment') else None
        )

        self.fit(
            dataset, max_epoch, train_batch_size, save_period,
            num_workers=train_batch_size
        )
        self.save(self.model_path)

        # self.load(self.model_path)
        # self.load(f'{self.model_dir}/{self.dataset_version}_last.pth')

        data = self.get_val_data()

        dataset = self.dataset(data, augment_func=self.val_data_augment)
        r = self.metric(
            dataset, predict_batch_size or train_batch_size,
            num_workers=predict_batch_size or train_batch_size
        )
        for k, v in r.items():
            self.logger.info({k: v})

    def params_search(self, model, var_params, const_params=dict(), **run_kwargs):
        def cur_run(**params):
            sub_version = ''
            for k, v in params.items():
                sub_version += f'{k}={v}'

            self.dataset_version = f'{self.dataset}/{sub_version}'
            self.logger = logging.getLogger()
            self.logger.info(f'{self.dataset_version = }')

            params = configs.merge_dict(params, const_params)
            configs.logger_init(log_dir=f'{self.model_dir}/{self.dataset_version}')
            self.model_path = f'{self.model_dir}/{self.dataset_version}/final.pth'
            self.save_result_dir = f'cache_data/{self.dataset_version}'
            self.model = model(**params)
            self.run(**run_kwargs)

        var_params = configs.permute_obj(var_params)
        for params in var_params:
            cur_run(**params)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def metric(self, *args, **kwargs):
        raise NotImplementedError

    def single_predict(self, image: np.ndarray, **kwargs):
        raise NotImplementedError

    def batch_predict(self, images: List[np.ndarray], batch_size=16, **kwargs):
        raise NotImplementedError

    def fragment_predict(self, image: np.ndarray, **kwargs):
        """predict large image. Tear picture to pieces to predict, and then merge the results"""
        raise NotImplementedError

    def data_augment(self, ret) -> dict:
        ret.update(Apply([
            # pixel_perturbation.MinMax(),
            channel.HWC2CHW()
        ])(**ret))
        return ret

    def val_data_augment(self, ret) -> dict:
        ret.update(Apply([
            # pixel_perturbation.MinMax(),
            channel.HWC2CHW()
        ])(**ret))
        return ret

    def val_data_restore(self, ret) -> dict:
        return ret

    def get_train_data(self, *args, **kwargs):
        raise NotImplementedError

    def get_val_data(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, save_path, save_type=MODEL, verbose=True):
        os_lib.mk_dir(Path(save_path).parent)

        if save_type == MODEL:
            torch.save(self.model, save_path)
        elif save_type == WEIGHT:
            torch.save(self.model.state_dict(), save_path)
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

    def load(self, save_path, save_type=MODEL, verbose=True):
        if save_type == MODEL:
            self.model = torch.load(save_path, map_location=self.device)
        elif save_type == WEIGHT:
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))
        elif save_type == JIT:
            self.model = torch.jit.load(save_path, map_location=self.device)
        else:
            raise ValueError(f'dont support {save_type = }')

        if verbose:
            self.logger.info(f'Successfully load {save_path} !')
