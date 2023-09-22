import os
from pathlib import Path
from .base import DataLoader, DataSaver, DataRegister, get_image, save_image
from utils import os_lib, converter
from zipfile import ZipFile
import cv2


class Loader(DataLoader):
    default_set_type = [DataRegister.MIX]

    def _call(self, task='original', **gen_kwargs):
        gen_func = Path(f'{self.data_dir}/{task}').glob(f'*.{self.image_suffix}')
        return self.gen_data(gen_func, task=task, **gen_kwargs)

    def get_ret(self, fp, image_type=DataRegister.PATH, **kwargs) -> dict:
        image_path = os.path.abspath(fp)
        image = get_image(image_path, image_type)

        return dict(
            _id=fp.name,
            image=image,
        )


class LoaderZip(DataLoader):
    default_set_type = [DataRegister.MIX]

    def _call(self, task='original', **gen_kwargs):
        f = ZipFile(f'{self.data_dir}/{task}.zip', 'r')
        gen_func = f.namelist()
        return self.gen_data(gen_func, task=task, f=f, **gen_kwargs)

    def get_ret(self, obj, image_type=DataRegister.PATH, f=None, **kwargs) -> dict:
        image = f.open(obj).read()
        image = converter.DataConvert.bytes_to_image(image)
        return dict(
            _id=obj,
            image=image,
        )


class Saver(DataSaver):
    def mkdirs(self, task='original', **kwargs):
        os_lib.mk_dir(f'{self.data_dir}/{task}')

    def _call(self, iter_data, **gen_kwargs):
        return self.gen_data(iter_data, **gen_kwargs)

    def parse_ret(self, ret, image_type=DataRegister.PATH, task='original', **get_kwargs):
        image = ret['image']
        _id = ret['_id']
        image_path = f'{self.data_dir}/{task}/{_id}'
        save_image(image, image_path, image_type)


class SaverZip(DataSaver):
    def mkdirs(self, **kwargs):
        os_lib.mk_dir(self.data_dir)

    def _call(self, iter_data, task='original', **gen_kwargs):
        f = ZipFile(f'{self.data_dir}/{task}.zip', 'w')
        return self.gen_data(iter_data, f=f, **gen_kwargs)

    def parse_ret(self, ret, sub_task='', f=None, **get_kwargs):
        image = ret['image']
        _id = ret['_id']
        image = converter.DataConvert.image_to_bytes(image)
        f.writestr(f'{sub_task}/{_id}', image)
