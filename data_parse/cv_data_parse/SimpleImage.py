import os
from pathlib import Path
from .base import DataLoader, DataSaver, DataRegister, get_image, save_image
from utils import os_lib


class Loader(DataLoader):
    default_set_type = [DataRegister.MIX]

    def _call(self, task='original', **kwargs):
        gen_func = Path(f'{self.data_dir}/{task}').glob(f'*.{self.image_suffix}')
        return self.gen_data(gen_func, task=task, **kwargs)

    def get_ret(self, fp, image_type=DataRegister.PATH, **kwargs) -> dict:
        image_path = os.path.abspath(fp)
        image = get_image(image_path, image_type)

        return dict(
            _id=fp.name,
            image=image,
        )


class Saver(DataSaver):
    def _call(self, iter_data, **gen_kwargs):
        return self.gen_data(iter_data, **gen_kwargs)

    def mkdirs(self, task='', **kwargs):
        os.mkdir(f'{self.data_dir}/{task}')

    def parse_ret(self, ret, image_type=DataRegister.PATH, task='', **get_kwargs):
        image = ret['image']
        _id = ret['_id']
        image_path = f'{self.data_dir}/{task}/{_id}'
        save_image(image, image_path, image_type)
