import os
from pathlib import Path
from typing import List
from .base import DataLoader, DataSaver, DataRegister, get_image, save_image
from utils import os_lib, converter, cv_utils


class Loader(DataLoader):
    """load a large image containing several pieces images"""
    default_set_type = [DataRegister.MIX]
    loader = os_lib.Loader(verbose=False)

    def _call(self, task='', **kwargs):
        gen_func = Path(f'{self.data_dir}/{task}').glob(f'*.{self.image_suffix}')
        return self.gen_data(gen_func, **kwargs)

    def get_ret(self, fp, size=None, **kwargs) -> List[dict]:
        image_path = os.path.abspath(fp)
        image = get_image(image_path, DataRegister.ARRAY)

        images = cv_utils.fragment_image(image, size)[0]

        for i, image in enumerate(images):
            yield dict(
                _id=f'{fp.stem}_{i}{fp.suffix}',
                image=image,
            )


class Saver(DataSaver):
    """save several pieces images as a large image file"""

    def mkdirs(self, task='', **kwargs):
        os_lib.mk_dir(f'{self.data_dir}/{task}')

    def _call(self, iter_data, **gen_kwargs):
        return self.gen_data(iter_data, **gen_kwargs)

    def parse_ret(self, rets, task='', save_label=False, **get_kwargs):
        if not len(rets):
            return

        images = [ret['image'] for ret in rets]
        image = cv_utils.splice_image(images)
        s_id = Path(rets[0]['_id'])  # _id fmt will like {i}{suffix}
        e_id = Path(rets[-1]['_id'])
        _id = f'{s_id.stem}_{e_id.stem}{s_id.suffix}'

        image_path = f'{self.data_dir}/{task}/{_id}'
        save_image(image, image_path, DataRegister.ARRAY)
