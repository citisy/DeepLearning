import os
import json
import cv2
import shutil
import numpy as np
from utils import os_lib
from .base import DataRegister, DataLoader, DataSaver, get_image, save_image
from .PaddleOcr_det import DataGenerator
from tqdm import tqdm
from pathlib import Path


class Loader(DataLoader):
    """https://github.com/PaddlePaddle/PaddleOCR

    Data structure:
        .
        ├── images
        │   └── [task]
        └── labels
            └── [set_task]
                  ├── train.txt  # per image file path per line
                  ├── test.txt   # would like to be empty or same to val.txt
                  └── val.txt

    """

    default_set_type = [DataRegister.TRAIN, DataRegister.TEST]
    image_suffix = 'png'

    def _call(self, set_type=DataRegister.TRAIN, image_type=DataRegister.TRAIN, set_task='', label_dir='labels', **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Returns:
            yield a dict had keys of
                _id: image file name
                image: see also image_type
                transcription: str

        Usage:
            .. code-block:: python

                # get data
                from cv_data_parse.PaddleOcr_rec import DataRegister, Loader

                loader = Loader('data/ppocr_icdar2015')
                train_data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY)[0]
                r = next(train_data)

                # visual
                from utils.visualize import ImageVisualize

                image = r['image']
                transcription = r['transcription']

        """

        with open(f'{self.data_dir}/{label_dir}/{set_task}/{set_type.value}.txt', 'r', encoding='utf8') as f:
            for line in f.read().strip().split('\n'):
                items = line.split('\t')
                if len(items) == 2:
                    image_paths, transcription = items
                    conf = 1
                elif len(items) == 3:
                    image_paths, transcription, conf = items
                else:
                    raise f'the line: {line}\nthere are {len(items)} items'

                # make sure that '[' is not in image_path
                if '[' in image_paths:
                    image_paths = eval(image_paths)
                else:
                    image_paths = [image_paths]

                for image_path in image_paths:
                    image_path = os.path.abspath(image_path)
                    image = get_image(image_path, image_type)

                    yield dict(
                        _id=Path(image_path).name,
                        image=image,
                        transcription=transcription,
                        conf=conf
                    )

    def load_dist_rec(self, set_type=DataRegister.TRAIN, image_type=DataRegister.TRAIN, set_task='', label_dir='labels', is_student=True, **kwargs):
        with open(f'{self.data_dir}/{label_dir}/{set_task}/{set_type.value}.txt', 'r', encoding='utf8') as f:
            for line in f.readlines():
                image_path, ret = line.split('\t')
                image_path = os.path.abspath(image_path)
                image = get_image(image_path, image_type)
                ret = json.loads(ret)
                if is_student:
                    ret = ret['Student']
                else:
                    ret = ret['Teacher']

                transcription = ret['label']

                yield dict(
                    _id=Path(image_path).name,
                    image=image,
                    transcription=transcription,
                )


class Saver(DataSaver):
    """https://github.com/PaddlePaddle/PaddleOCR

    Data structure:
        .
        ├── images
        │   └── [task]
        └── labels
            └── [set_task]
                  ├── train.txt  # per image file path per line
                  ├── test.txt   # would like to be empty or same to val.txt
                  └── val.txt

    """

    def mkdirs(self, set_types, **kwargs):
        task = kwargs.get('task', '')
        set_task = kwargs.get('set_task', '')

        os_lib.mk_dir(f'{self.data_dir}/images/{task}')
        os_lib.mk_dir(f'{self.data_dir}/labels/{set_task}')

    def _call(self, iter_data, set_type, image_type, task='', set_task='', **kwargs):
        """

        Args:
            iter_data:
            set_type:
            image_type:
            task:
            set_task:
            **kwargs:

        Returns:

        Usage:
            .. code-block:: python

                # save rec test data to another dir
                loader = Loader('your load data dir')
                iter_data = loader(set_type=DataRegister.TEST, image_type=DataRegister.PATH, generator=True, set_task='rec labels dir')

                saver = Saver('your save data dir')
                saver.save_rec(iter_data, set_type=DataRegister.TEST, image_type=DataRegister.PATH, task='rec images dir', set_task='rec labels dir')

        """
        f = open(f'{self.data_dir}/labels/{set_task}/{set_type.value}.txt', 'w', encoding='utf8')

        for dic in tqdm(iter_data):
            image = dic['image']
            transcription = dic['transcription']
            _id = dic['_id']

            image_path = os.path.abspath(f'{self.data_dir}/images/{task}/{_id}')
            save_image(image, image_path, image_type)
            f.write(f'{image_path}\t{transcription}\n')

        f.close()

    def save_rec_from_det(self, iter_data, set_type, *args, task='', set_task='', **kwargs):
        """

        Args:
            iter_data:
            set_type:
            task:
            set_task:

        Returns:

        Usage:
            .. code-block:: python

                loader = Loader('your load data dir')
                data = loader(set_type=DataRegister.ALL, image_type=DataRegister.ARRAY, generator=True, set_task='det labels dir')

                saver = Saver('your save data dir')
                for iter_data, set_type in zip(data, loader.default_set_type):
                    saver.save_rec_from_det(iter_data, set_type=set_type, task='rec images dir', set_task='rec labels dir')

        """
        os_lib.mk_dir(f'{self.data_dir}/labels/{set_task}')
        os_lib.mk_dir(f'{self.data_dir}/images/{task}')
        f = open(f'{self.data_dir}/labels/{set_task}/{set_type.value}.txt', 'w', encoding='utf8')

        for dic in tqdm(iter_data):
            image = dic['image']
            transcriptions = dic['transcriptions']
            segmentations = dic['segmentations']
            _id = dic['_id']

            for i, (segmentation, transcription) in enumerate(zip(segmentations, transcriptions)):
                x1 = np.min(segmentation[:, 0], axis=0)
                x2 = np.max(segmentation[:, 0], axis=0)
                y1 = np.min(segmentation[:, 1], axis=0)
                y2 = np.max(segmentation[:, 1], axis=0)

                img = image[y1: y2, x1:x2]

                p = Path(_id)
                img_fp = f'{self.data_dir}/images/{task}/{p.stem}_{i}{p.suffix}'
                cv2.imwrite(img_fp, img)

                f.write(f'{img_fp}\t{transcription.strip()}\n')

        f.close()

