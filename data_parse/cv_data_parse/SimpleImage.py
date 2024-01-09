import os
import json
from pathlib import Path
from .base import DataLoader, DataSaver, DataRegister, get_image, save_image
from utils import os_lib, converter
from zipfile import ZipFile


class Loader(DataLoader):
    default_set_type = [DataRegister.MIX]
    loader = os_lib.Loader(verbose=False)

    def _call(self, task='original', **gen_kwargs):
        gen_func = Path(f'{self.data_dir}/{task}').glob(f'*.{self.image_suffix}')
        return self.gen_data(gen_func, task=task, **gen_kwargs)

    def get_ret(self, fp, image_type=DataRegister.PATH, return_label=False, **kwargs) -> dict:
        image_path = os.path.abspath(fp)
        image = get_image(image_path, image_type)

        label = {}
        if return_label:
            label_path = str(fp).replace(self.image_suffix, 'json')
            label = self.loader.load_json(label_path)

        return dict(
            _id=fp.name,
            image=image,
            **label
        )


class ZipLoader(DataLoader):
    default_set_type = [DataRegister.MIX]

    def _call(self, task='original', **gen_kwargs):
        zip_file = ZipFile(f'{self.data_dir}/{task}.zip', 'r')
        gen_func = zip_file.namelist()
        return self.gen_data(gen_func, task=task, zip_file=zip_file, **gen_kwargs)

    def get_ret(self, obj, image_type=DataRegister.PATH, zip_file=None, return_label=False, **kwargs) -> dict:
        image = zip_file.open(obj).read()
        image = converter.DataConvert.bytes_to_image(image)

        label = {}
        if return_label:
            label_path = str(obj).replace(self.image_suffix, 'json')
            label = json.loads(str(zip_file.open(label_path).read()))

        return dict(
            _id=obj,
            image=image,
            **label
        )


class Saver(DataSaver):
    saver = os_lib.Saver(verbose=False)

    def mkdirs(self, task='original', **kwargs):
        os_lib.mk_dir(f'{self.data_dir}/{task}')

    def _call(self, iter_data, **gen_kwargs):
        return self.gen_data(iter_data, **gen_kwargs)

    def parse_ret(self, ret, image_type=DataRegister.PATH, task='original', save_label=False, **get_kwargs):
        image = ret['image']
        _id = ret['_id']
        image_path = f'{self.data_dir}/{task}/{_id}'
        save_image(image, image_path, image_type)

        if save_label:
            ret = {k: v for k, v in ret if k != 'image'}
            label_path = image_path.replace(Path(image_path).suffix, '.json')
            self.saver.save_json(ret, label_path)


class ZipSaver(DataSaver):
    def mkdirs(self, **kwargs):
        os_lib.mk_dir(self.data_dir)

    def _call(self, iter_data, task='original', **gen_kwargs):
        f = ZipFile(f'{self.data_dir}/{task}.zip', 'w')
        return self.gen_data(iter_data, f=f, **gen_kwargs)

    def parse_ret(self, ret, sub_task='', f=None, save_label=False, **get_kwargs):
        image = ret['image']
        _id = ret['_id']
        image = converter.DataConvert.image_to_bytes(image)
        image_path = f'{sub_task}/{_id}'
        f.writestr(image_path, image)

        if save_label:
            ret = {k: v for k, v in ret if k != 'image'}
            label_path = image_path.replace(Path(image_path).suffix, '.json')
            f.writestr(label_path, json.dumps(ret))
