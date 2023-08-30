import os
import json
import numpy as np
from utils import os_lib
from .base import DataRegister, DataLoader, DataSaver, DatasetGenerator, get_image, save_image
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
            from data_parse.cv_data_parse.PaddleOcr_det import DataRegister, Loader

            loader = Loader('data/ppocr_icdar2015')
            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.ARRAY)
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

    def _call(self, set_type=DataRegister.TRAIN, set_task='', label_dir='labels', **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Returns:
            yield a dict had keys of
                _id: image file name
                image: see also image_type
                segmentations: a np.ndarray with shape of (-1, 4, 2)
                transcriptions: List[str]

        Usage:
            .. code-block:: python

                # get data
                from data_parse.data_parse.cv_data_parse.PaddleOcr import DataRegister, Loader

                loader = Loader('data/ppocr_icdar2015')
                train_data = loader.load_det(set_type=DataRegister.TRAIN, image_type=DataRegister.ARRAY)
                r = next(train_data)

                # visual
                from utils.visualize import ImageVisualize

                image = r['image']
                segmentations = r['segmentations']
                transcriptions = r['transcriptions']

                vis_image = np.zeros_like(image) + 255
                vis_image = ImageVisualize.box(vis_image, segmentations)
                vis_image = ImageVisualize.text(vis_image, segmentations, transcriptions)

        """

        with open(f'{self.data_dir}/{label_dir}/{set_task}/{set_type.value}.txt', 'r', encoding='utf8') as f:
            gen_func = f.read().strip().split('\n')
        return self.gen_data(gen_func, **kwargs)

    def get_ret(self, line, image_type=DataRegister.PATH, **kwargs) -> dict:
        image_path, labels = line.split('\t', 1)
        image_path = os.path.abspath(image_path)
        image = get_image(image_path, image_type)

        labels = json.loads(labels)
        segmentations, transcriptions = [], []
        for label in labels:
            segmentations.append(label['points'])  # (-1, 4, 2)
            transcriptions.append(label['transcription'])  # (-1, #str)

        segmentations = np.array(segmentations)
        bboxes = np.zeros((len(segmentations), 4))
        bboxes[:, :2] = np.min(segmentations, axis=1)
        bboxes[:, -2:] = np.max(segmentations, axis=1)

        return dict(
            _id=Path(image_path).name,
            image=image,
            segmentations=segmentations,
            bboxes=bboxes,
            transcriptions=transcriptions,
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
            from data_parse.cv_data_parse.Icdar import Loader
            from utils.register import DataRegister

            loader = Loader('data/ICDAR2015')
            data = loader(set_type=DataRegister.TRAIN, image_type=DataRegister.FILE, generator=True)

            # save as PaddleOcr type
            from cv_data_parse.PaddleOcr_det import Saver

            saver = Saver('data/ppocr_icdar2015')
            saver(data, set_type=DataRegister.TRAIN, image_type=DataRegister.PATH)
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

        Returns:

        Usage:
            .. code-block:: python

                # save det test data to another dir
                loader = Loader('your load data dir')
                iter_data = loader.load_det(set_type=DataRegister.TEST, image_type=DataRegister.PATH, generator=True, set_task='det labels dir')

                iter_data = Saver('your save data dir')
                saver.save_det(iter_data, set_type=DataRegister.TEST, image_type=DataRegister.PATH, task='det images dir', set_task='det labels dir')

        """
        f = open(f'{self.data_dir}/labels/{set_task}/{set_type.value}.txt', 'w', encoding='utf8')

        for dic in iter_data:
            image = dic['image']
            segmentations = np.array(dic['segmentations']).tolist()
            transcriptions = dic['transcriptions']
            _id = dic['_id']

            image_path = os.path.abspath(f'{self.data_dir}/images/{task}/{_id}')
            save_image(image, image_path, image_type)
            labels = [{'transcription': t, 'points': s} for t, s in zip(transcriptions, segmentations)]

            f.write(f'{image_path}\t{json.dumps(labels, ensure_ascii=False)}\n')

        f.close()


class Generator(DatasetGenerator):
    image_suffix = 'png'
    label_suffix = 'txt'

    def gen_sets(
            self, label_files=(), save_dir='', set_task='',
            id_distinguish='', id_sort=False,
            set_names=('train', 'test'), split_ratio=(0.8, 1), **kwargs
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

                from data_parse.cv_data_parse.PaddleOcr_det import Generator

                # simple
                # data_dir for save_dir
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
