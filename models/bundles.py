import os
import warnings
from pathlib import Path

from utils import torch_utils


class Config:
    default_model = ''

    @classmethod
    def make_full_config(cls) -> dict:
        raise NotImplemented

    @classmethod
    def get(cls, name=None):
        config_dict = cls.make_full_config()
        if name not in config_dict:
            warnings.warn(f'config `{name}` not in current config dict, the whole config keys is {list(config_dict.keys())}, '
                          f'use default `{cls.default_model}` config now, please check about.')
            name = cls.default_model
        return config_dict[name]


class WeightLoader:
    @classmethod
    def from_hf(cls, save_path, save_name='pytorch_model.bin', **kwargs):
        return cls.auto_load(save_path, save_name, **kwargs)

    @staticmethod
    def get_file_name(save_path, save_name='', **kwargs):
        if os.path.isfile(save_path):
            file_name = save_path
        elif os.path.isfile(f'{save_path}/{save_name}'):
            file_name = f'{save_path}/{save_name}'
        else:
            raise ValueError(f'can not find file in {save_path} and {save_name}')
        return file_name

    @classmethod
    def auto_load(cls, save_path: str | list, save_name: str | list = '', suffix='', **kwargs):
        if not isinstance(save_path, (str, Path)):
            state_dict = {}
            for _save_path in save_path:
                state_dict.update(cls.auto_load(_save_path, save_name=save_name, suffix=suffix, **kwargs))

        elif not isinstance(save_name, (str, Path)):
            state_dict = {}
            for _save_name in save_name:
                state_dict.update(cls.auto_load(save_path, save_name=_save_name, suffix=suffix, **kwargs))

        elif os.path.isfile(f'{save_path}/{save_name}'):
            state_dict = torch_utils.Load.from_file(f'{save_path}/{save_name}', **kwargs)

        elif os.path.isfile(save_path):
            state_dict = torch_utils.Load.from_file(save_path, **kwargs)

        elif os.path.isdir(save_path):
            state_dict = torch_utils.Load.from_dir(save_path, suffix, **kwargs)

        else:
            state_dict = cls.auto_download(save_path, save_name=save_name, **kwargs)

        return state_dict

    @classmethod
    def auto_download(cls, save_path, **kwargs):
        raise NotImplemented
