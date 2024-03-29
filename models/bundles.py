import os
import torch
import warnings


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
    def auto_load(cls, save_path, save_name='', **kwargs):
        try:
            file_name = cls.get_file_name(save_path, save_name, **kwargs)
            state_dict = torch.load(file_name)
        except ValueError:
            state_dict = cls.auto_download(save_path, save_name=save_name, **kwargs)
        return state_dict

    @classmethod
    def auto_download(cls, save_path, **kwargs):
        raise NotImplemented
