import os
import json
import cv2
import shutil
import numpy as np
from utils import os_lib
from cv_data_parse.base import DataRegister, DataLoader, DataSaver, DataGenerator
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
                    segmentation.append(label['points'])  # (-1, 4, 2)
                    transcription.append(label['transcription'])  # (-1, #str)

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


class Generator(DataGenerator):
    image_suffix = 'png'
    label_suffix = 'txt'

    def gen_sets(
            self, label_files=(), save_dir='', set_task='',
            id_distinguish='', id_sort=False,
            set_names=('train', 'test'), split_ratio=(0.8, 1)
    ):
        """generate new set for training or testing

        Args:
            label_files(None or tuple): special paths of label file or travel file in Generator.label_dir
            save_dir: special dir or use {Generator.data_dir}/labels/{task}
            set_task:
            id_distinguish:
                the image file name where having same id must not be split to different sets.
                e.g. id_distinguish = '_', the fmt of image file name would like '{id}_{sub_id}.png'
            id_sort: if id_distinguish is set, True for sorting by sub_ids and sub_ids must be int type
            set_names: save txt name
            split_ratio:
                split ratio for each set, the shape must apply for set_names
                if id_distinguish is set, the ration is num of ids not files

        Data structure:
            .
            └── labels
                └── [set_task]
                      ├── train.txt  # per image file path per line
                      ├── test.txt   # would like to be empty or same to val.txt
                      └── val.txt

        Usage:
            .. code-block:: python

                from cv_data_parse.PaddleOcr import Generator

                # simple
                gen = Generator(data_dir='data/ppocr_icdar2015', label_dir='data/ppocr_icdar2015/labels/0')
                gen.gen_sets(set_task='1')

                # special path
                gen = Generator()
                gen.gen_sets(
                    label_files=('data/ocr0/labels/train.txt', 'data/ocr1/labels/train.txt'),
                    save_dir='data/ocr2/labels/1'
                )

        """
        save_dir = save_dir or f'{self.data_dir}/labels/{set_task}'
        os_lib.mk_dir(save_dir)

        if not label_files:
            label_files = [i for i in Path(self.label_dir).glob(f'*.{self.label_suffix}')]

        data = []
        idx = []
        for label_file in label_files:
            with open(label_file, 'r', encoding='utf8') as f:
                for line in f.read().strip().split('\n'):
                    data.append(line)
                    if id_distinguish:
                        i, _ = line.split('\t', 1)
                        idx.append(i)

        self._gen_sets(data, idx, id_distinguish, id_sort, save_dir, set_names, split_ratio)
