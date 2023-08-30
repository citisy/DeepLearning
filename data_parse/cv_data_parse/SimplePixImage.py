import os
from pathlib import Path
from .base import DataLoader, DataSaver, DataRegister, get_image, save_image
from utils import os_lib


class Loader(DataLoader):
    """a simple loader for image segmentation, image generation and so on

    Data structure:
        .
        ├── [task]
        └── [pix_task]

    Usage:
        .. code-block:: python

            # get data
            from data_parse.cv_data_parse.SimplePixImage import DataRegister, Loader, DataVisualizer

            data_dir = 'data/simple_pix_image'
            loader = Loader(data_dir)
            data = loader(generator=True, image_type=DataRegister.ARRAY)

            # visual train dataset
            DataVisualizer(f'{data_dir}/visuals', verbose=False)(data[0])

    """
    default_set_type = [DataRegister.MIX]

    def _call(self, task='original', **kwargs):
        gen_func = Path(f'{self.data_dir}/{task}').glob(f'*.{self.image_suffix}')
        return self.gen_data(gen_func, task=task, **kwargs)

    def get_ret(self, fp, image_type=DataRegister.PATH, task='original', pix_task='pixels', **kwargs) -> dict:
        image_path = os.path.abspath(fp)
        image = get_image(image_path, image_type)

        pix_image_path = image_path.replace(task, pix_task)
        pix_image = get_image(pix_image_path, image_type)

        return dict(
            _id=fp.name,
            image=image,
            pix_image=pix_image,
        )


class Saver(DataSaver):
    """https://github.com/ultralytics/yolov5

    Data structure:
        .
        ├── [task]
        └── [pix_task]

    Usage:
        .. code-block:: python

            # convert cmp_facade to SimplePixImage
            # load data from cmp_facade
            from data_parse.cv_data_parse.cmp_facade import Loader
            from utils.register import DataRegister
            loader = Loader('data/cmp_facade')
            data = loader()

            # save as SimplePixImage type
            from data_parse.cv_data_parse.SimplePixImage import Saver
            saver = Saver('data/simple_pix_image')
            saver(data)

    """

    def mkdirs(self, set_types, task='', pix_task='', **kwargs):
        os_lib.mk_dir(f'{self.data_dir}/{task}')
        os_lib.mk_dir(f'{self.data_dir}/{pix_task}')

    def _call(self, iter_data, image_type=DataRegister.PATH, task='', pix_task='', **kwargs):
        for ret in iter_data:
            ret = self.convert_func(ret)

            image = ret['image']
            pix_image = ret['pix_image']
            _id = ret['_id']

            image_path = f'{self.data_dir}/{task}/{_id}'
            pix_image_path = image_path.replace(task, pix_task)
            save_image(image, image_path, image_type)
            save_image(pix_image, pix_image_path, image_type)
