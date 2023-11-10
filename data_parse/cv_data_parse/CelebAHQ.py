import os
import cv2
import pandas as pd
from zipfile import ZipFile
from pathlib import Path
from utils import cv_utils, converter
from .base import DataLoader, DataRegister, get_image
import io


class Loader(DataLoader):
    """http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html
    Data structure:
        .
        ├── CelebA-HQ-to-CelebA-mapping.txt
        ├── CelebAMask-HQ-attribute-anno.txt
        ├── CelebAMask-HQ-pose-anno.txt
        ├── CelebAMask-HQ-mask-anno
        │   └── [i]
        └── CelebA-HQ-img
    """
    image_suffix = 'jpg'

    def _call(self, **gen_kwargs):
        gen_func = Path(f'{self.data_dir}/CelebA-HQ-img').glob(f'*{self.image_suffix}')

        return self.gen_data(gen_func, **gen_kwargs)

    def get_ret(self, _id, image_type=DataRegister.ARRAY, only_image=False, **kwargs) -> dict:
        image_path = f'CelebA-HQ-img/{_id}'
        image = get_image(image_path, image_type)

        ret = dict(
            _id=_id,
            image=image,
        )

        return ret


class ZipLoader(Loader):
    """http://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html
    Data structure:
        ./CelebAMask-HQ
        ├── CelebA-HQ-to-CelebA-mapping.txt
        ├── CelebAMask-HQ-attribute-anno.txt
        ├── CelebAMask-HQ-pose-anno.txt
        ├── CelebAMask-HQ-mask-anno
        │   └── [i]
        └── CelebA-HQ-img
    """

    def _call(self, **gen_kwargs):
        zip_file = ZipFile(f'{self.data_dir}', 'r')

        def gen():
            for fn in zip_file.namelist():
                if Path(fn).suffix == '.jpg':
                    yield fn

        return self.gen_data(gen(), zip_file=zip_file, **gen_kwargs)

    def get_ret(self, p, only_image=False, zip_file=None, **kwargs) -> dict:
        _id = Path(p).name
        image = zip_file.open(p).read()
        image = converter.DataConvert.bytes_to_image(image)

        ret = dict(
            _id=_id,
            image=image,
        )

        return ret
