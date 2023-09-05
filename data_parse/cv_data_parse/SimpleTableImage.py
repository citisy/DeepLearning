import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from .base import DataLoader, DataSaver, DataRegister, get_image, save_image
from utils import os_lib, converter


class Loader(DataLoader):
    default_set_type = [DataRegister.MIX]
    loader = os_lib.Loader(verbose=False)
    image_suffix = 'png'

    def _call(self, task='original', **kwargs):
        gen_func = Path(f'{self.data_dir}/{task}').glob(f'*.{self.image_suffix}')
        return self.gen_data(gen_func, **kwargs)

    def get_ret(self, fp, image_type=DataRegister.PATH, **kwargs) -> dict:
        image_path = os.path.abspath(fp)
        image = get_image(image_path, image_type)

        label_path = str(fp).replace(self.image_suffix, 'json')
        label = self.loader.load_json(label_path)
        return dict(
            _id=fp.name,
            image=image,
            **label
        )
