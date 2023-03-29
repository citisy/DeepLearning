import os
import json
import cv2
import shutil
import numpy as np
from utils import os_lib
from cv_data_parse.base import DataRegister, DataLoader, DataSaver, DataGenerator
from tqdm import tqdm
from pathlib import Path

DET = 'det'
REC = 'rec'


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
            data = loader(set_type=DataRegister.ALL, generator=True, image_type=DataRegister.IMAGE)
            r = next(data[0])

            # visual
            from utils.visualize import ImageVisualize

            image = r['image']
            segmentations = r['segmentations']
            transcriptions = r['transcriptions']

            vis_image = np.zeros_like(image) + 255
            vis_image = ImageVisualize.box(vis_image, segmentations)
            vis_image = ImageVisualize.text(vis_image, segmentations, transcriptions)
    """

    default_set_type = [DataRegister.TRAIN, DataRegister.TEST]
    image_suffix = 'png'

    def _call(self, set_type, image_type, set_task='', train_task=DET, **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Args:
            set_type:
            image_type:
            set_task(str): one of dir name in `labels` dir
            train_task(str): 'det' or 'rec'

        Returns:
            see also `self.load_det` if train_task='det'
            see also `self.load_rec` if train_task='rec'

        """

        if train_task == DET:
            return self.load_det(set_type, image_type, set_task=set_task, **kwargs)
        elif train_task == REC:
            return self.load_rec(set_type, image_type, set_task=set_task, **kwargs)
        else:
            raise ValueError(f'dont support {train_task = }')

    def load_det(self, set_type, image_type, set_task='', **kwargs):
        """See Also `self._call`

        Returns:
            a dict had keys of
                _id: image file name
                image: see also image_type
                segmentations: a np.ndarray with shape of (-1, 4, 2)
                transcriptions: List[str]
        """

        with open(f'{self.data_dir}/labels/{set_task}/{set_type.value}.txt', 'r', encoding='utf8') as f:
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
                segmentations, transcriptions = [], []
                for label in labels:
                    segmentations.append(label['points'])  # (-1, 4, 2)
                    transcriptions.append(label['transcription'])  # (-1, #str)

                segmentations = np.array(segmentations)

                yield dict(
                    _id=Path(image_path).name,
                    image=image,
                    segmentations=segmentations,
                    transcriptions=transcriptions,
                )

    def load_rec(self, set_type, image_type, set_task='', **kwargs):
        """See Also `self._call`

        Returns:
            a dict had keys of
                _id: image file name
                image: see also image_type
                transcription: str
        """

        with open(f'{self.data_dir}/labels/{set_task}/{set_type.value}.txt', 'r', encoding='utf8') as f:
            for line in f.read().strip().split('\n'):
                image_paths, transcription = line.split('\t', 1)

                if '[' in image_path:
                    image_paths = eval(image_paths)
                else:
                    image_paths = [image_paths]

                for image_path in image_paths:
                    image_path = os.path.abspath(image_path)
                    if image_type == DataRegister.PATH:
                        image = image_path
                    elif image_type == DataRegister.IMAGE:
                        image = cv2.imread(image_path)
                    else:
                        raise ValueError(f'Unknown input {image_type = }')

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

    Usage:
        .. code-block:: python

            # convert ICDAR2015 to PaddleOcr
            # load data from ICDAR2015
            from cv_data_parse.Icdar import Loader
            from utils.register import DataRegister

            loader = Loader('data/ICDAR2015')
            data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.FILE, generator=True)

            # save as PaddleOcr type
            from cv_data_parse.PaddleOcr import Saver

            saver = Saver('data/ppocr_icdar2015')
            saver(data, set_type=DataRegister.TRAIN, image_type=DataRegister.FILE)
    """

    def mkdirs(self, set_types, **kwargs):
        task = kwargs.get('task', '')
        set_task = kwargs.get('set_task', '')

        os_lib.mk_dir(f'{self.data_dir}/images/{task}')
        os_lib.mk_dir(f'{self.data_dir}/labels/{set_task}')

    def _call(self, iter_data, set_type, image_type, train_task=DET, **kwargs):
        if train_task == DET:
            return self.save_det(iter_data, set_type, image_type, **kwargs)
        elif train_task == REC:
            return self.save_rec(iter_data, set_type, image_type, **kwargs)
        else:
            raise ValueError(f'dont support {train_task = }')

    def save_det(self, iter_data, set_type, image_type, **kwargs):
        task = kwargs.get('task', '')
        set_task = kwargs.get('set_task', '')

        f = open(f'{self.data_dir}/labels/{set_task}/{set_type.value}.txt', 'w', encoding='utf8')

        for dic in tqdm(iter_data):
            image = dic['image']
            segmentations = np.array(dic['segmentations']).tolist()
            transcriptions = dic['transcriptions']
            _id = dic['_id']

            image_path = os.path.abspath(f'{self.data_dir}/images/{task}/{_id}')

            if image_type == DataRegister.PATH:
                shutil.copy(image, image_path)
            elif image_type == DataRegister.IMAGE:
                cv2.imwrite(image_path, image)
            else:
                raise ValueError(f'Unknown input {image_type = }')

            labels = [{'transcription': t, 'points': s} for t, s in zip(transcriptions, segmentations)]

            f.write(f'{image_path}\t{json.dumps(labels, ensure_ascii=False)}\n')

        f.close()

    def save_rec(self, iter_data, set_type, image_type, **kwargs):
        task = kwargs.get('task', '')
        set_task = kwargs.get('set_task', '')

        f = open(f'{self.data_dir}/labels/{set_task}/{set_type.value}.txt', 'w', encoding='utf8')

        for dic in tqdm(iter_data):
            image = dic['image']
            transcription = dic['transcription']
            _id = dic['_id']

            image_path = os.path.abspath(f'{self.data_dir}/images/{task}/{_id}')

            if image_type == DataRegister.PATH:
                shutil.copy(image, image_path)
            elif image_type == DataRegister.IMAGE:
                cv2.imwrite(image_path, image)
            else:
                raise ValueError(f'Unknown input {image_type = }')

            f.write(f'{image_path}\t{transcription}\n')

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
