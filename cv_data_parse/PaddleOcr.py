import os
import json
import cv2
import shutil
import numpy as np
from utils import os_lib
from cv_data_parse.base import DataRegister, DataLoader, DataSaver
from pathlib import Path
from tqdm import tqdm


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
    Usage:
        .. code-block:: python

            # get data
            from cv_data_parse.PaddleOcr import DataRegister, Loader

            loader = Loader('data/ppocr_icdar2015')
            data = loader(data_type=DataRegister.ALL, generator=True, image_type=DataRegister.IMAGE)
            r = next(data[0])

            # visual
            from utils.visualize import ImageVisualize

            image = r['image']
            segmentation = r['segmentation']
            transcription = r['transcription']

            vis_image = np.zeros_like(image) + 255
            vis_image = ImageVisualize.box(vis_image, segmentation)
            vis_image = ImageVisualize.text(vis_image, segmentation, transcription)
    """

    default_load_type = [DataRegister.TRAIN, DataRegister.TEST]
    image_suffix = 'png'

    def _call(self, load_type, image_type, set_task='', **kwargs):
        with open(f'{self.data_dir}/labels/{set_task}/{load_type.value}.txt', 'r', encoding='utf8') as f:
            for line in f.read().strip().split('\n'):
                image_path, labels = line.split('\t', 1)

                image_path = os.path.abspath(image_path)
                if image_type == DataRegister.PATH:
                    image = image_path
                elif image_type == DataRegister.IMAGE:
                    image = cv2.imread(image_path)
                else:
                    raise ValueError(f'Unknown input {image_type = }')

                labels = json.loads(labels)
                segmentation, transcription = [], []
                for label in labels:
                    segmentation.append(label['points'])            # (-1, 4, 2)
                    transcription.append(label['transcription'])    # (-1, #str)

                segmentation = np.array(segmentation)

                yield dict(
                    _id=Path(image_path).name,
                    image=image,
                    segmentation=segmentation,
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
    Usage:
        .. code-block:: python

            # convert ICDAR2015 to PaddleOcr
            # load data from ICDAR2015
            from cv_data_parse.Icdar import Loader
            from utils.register import DataRegister

            loader = Loader('data/ICDAR2015')
            data = loader(data_type=DataRegister.TRAIN, image_type=DataRegister.FILE, generator=True)

            # save as PaddleOcr type
            from cv_data_parse.PaddleOcr import Saver

            saver = Saver('data/ppocr_icdar2015')
            saver(data, data_type=DataRegister.TRAIN, image_type=DataRegister.FILE)
    """

    def __call__(self, data, data_type=DataRegister.ALL, image_type=DataRegister.PATH, **kwargs):
        task = kwargs.get('task', '')
        set_task = kwargs.get('set_task', '')

        os_lib.mk_dir(f'{self.data_dir}/images/{task}')
        os_lib.mk_dir(f'{self.data_dir}/labels/{set_task}')

        super().__call__(data, data_type, image_type, **kwargs)

    def _call(self, iter_data, load_type, image_type, **kwargs):
        task = kwargs.get('task', '')
        set_task = kwargs.get('set_task', '')

        f = open(f'{self.data_dir}/labels/{set_task}/{load_type.value}.txt', 'w', encoding='utf8')

        for dic in tqdm(iter_data):
            image = dic['image']
            segmentation = np.array(dic['segmentation']).tolist()
            transcription = dic['transcription']
            _id = dic['_id']

            image_path = os.path.abspath(f'{self.data_dir}/images/{task}/{_id}')

            if image_type == DataRegister.PATH:
                shutil.copy(image, image_path)
            elif image_type == DataRegister.IMAGE:
                cv2.imwrite(image_path, image)
            else:
                raise ValueError(f'Unknown input {image_type = }')

            labels = [{'transcription': t, 'points': s} for s, t in zip(segmentation, transcription)]

            f.write(f'{image_path}\t{json.dumps(labels, ensure_ascii=False)}\n')

        f.close()
