import cv2
import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from utils import os_lib, converter
from cv_data_parse.base import DataRegister, DataLoader, DataSaver, DataGenerator


class Loader(DataLoader):
    """https://github.com/ultralytics/yolov5

    Data structure:
        .
        ├── images
        │   └── [task]
        ├── labels
        │   └── [task]
        └── image_sets
            └── [set_task]
                  ├── train.txt  # per image file path per line
                  ├── test.txt   # would like to be empty or same to val.txt
                  └── val.txt

    Usage:
        .. code-block:: python

            # get data
            from cv_data_parse.YoloV5 import DataRegister, Loader

            loader = Loader('data/Yolov5Data')
            data = loader(data_type=DataRegister.ALL, generator=True, image_type=DataRegister.IMAGE)
            r = next(data[0])

            # visual
            from utils.visualize import ImageVisualize

            image = r['image']
            bboxes = r['bboxes']
            classes = r['classes']
            classes = [loader.classes[_] for _ in classes]
            image = ImageVisualize.label_box(image, bboxes, classes, line_thickness=2)

    """

    image_suffix = 'png'

    def _call(self, load_type, image_type, task='', set_task='', **kwargs):
        if load_type == DataRegister.place_holder:
            return self.load_total(image_type, task, **kwargs)
        else:
            return self.load_set(load_type, image_type, set_task, **kwargs)

    def load_total(self, image_type, task, **kwargs):
        for img_fp in Path(f'{self.data_dir}/images/{task}').glob(f'*.{self.image_suffix}'):
            image_path = os.path.abspath(img_fp)
            if image_type == DataRegister.PATH:
                image = image_path
            elif image_type == DataRegister.IMAGE:
                image = cv2.imread(image_path)
            else:
                raise ValueError(f'Unknown input {image_type = }')

            labels = np.genfromtxt(image_path.replace('images', 'labels').replace(f'.{self.image_suffix}', '.txt')).reshape((-1, 5))

            # norm bboxes, (mid x, mid y, w, h)
            bboxes = labels[:, 1:]
            classes = labels[:, 0].astype(int)

            yield dict(
                _id=img_fp.name,
                image=image,
                bboxes=bboxes,
                classes=classes,
            )

    def load_set(self, load_type, image_type, set_task, **kwargs):
        with open(f'{self.data_dir}/image_sets/{set_task}/{load_type.value}.txt', 'r', encoding='utf8') as f:
            for line in f.read().split('\n'):
                image_path = os.path.abspath(line)
                if image_type == DataRegister.PATH:
                    image = image_path
                elif image_type == DataRegister.IMAGE:
                    image = cv2.imread(image_path)
                else:
                    raise ValueError(f'Unknown input {image_type = }')

                labels = np.genfromtxt(image_path.replace('images', 'labels').replace(f'.{self.image_suffix}', '.txt')).reshape((-1, 5))

                # norm bboxes, (mid x, mid y, w, h)
                bboxes = labels[:, 1:]
                classes = labels[:, 0]

                yield dict(
                    _id=Path(line).name,
                    image=image,
                    bboxes=bboxes,
                    classes=classes,
                )

    def load_full_labels(self, task):
        """format of saved label txt like (class, conf, w, h, c, x1, y1, x2, y2)
        only return labels but no images. can use _id to load images"""
        for fp in Path(f'{self.data_dir}/full_labels/{task}').glob('*.txt'):
            # (class_, conf, w, h, x1, y1, x2, y2)
            labels = np.genfromtxt(fp)
            yield dict(
                _id=fp.name.replace('.txt', '.png'),
                task=task,
                classes=labels[:, 0],
                confs=labels[:, 1],
                image_shape=labels[:, 2:5],  # (w, h, c)
                bboxes=labels[:, 5:10],
            )


class Saver(DataSaver):
    """https://github.com/ultralytics/yolov5

    Data structure:
        .
        ├── images
        │   └── [task]
        ├── labels
        │   └── [task]
        └── image_sets
            └── [set_task]
                  ├── train.txt  # per image file path per line
                  ├── test.txt   # would like to be empty or same to val.txt
                  └── val.txt

    Usage:
        .. code-block:: python

            # convert voc to yolov5
            # load data from voc
            from cv_data_parse.Voc import Loader
            from utils.register import DataRegister
            loader = Loader('data/VOC2012')
            data = loader(data_type=DataRegister.TRAIN)

            # save as yolov5 type
            from cv_data_parse.YoloV5 import Saver
            saver = Saver('data/Yolov5')
            saver(data, data_type=DataRegister.TRAIN)

    """

    def __call__(self, data, data_type=DataRegister.ALL, image_type=DataRegister.PATH, **kwargs):
        task = kwargs.get('task', '')
        set_task = kwargs.get('set_task', '')

        os_lib.mk_dir(f'{self.data_dir}/image_sets/{set_task}')
        os_lib.mk_dir(f'{self.data_dir}/images/{task}')
        os_lib.mk_dir(f'{self.data_dir}/labels/{task}')

        super().__call__(data, data_type, image_type, **kwargs)

    def _call(self, iter_data, load_type, image_type, **kwargs):
        task = kwargs.get('task', '')
        set_task = kwargs.get('set_task', '')

        if load_type == DataRegister.place_holder:
            f = os_lib.EmptyOs()
        else:
            f = open(f'{self.data_dir}/image_sets/{set_task}/{load_type.value}.txt', 'w', encoding='utf8')

        for dic in tqdm(iter_data):
            image = dic['image']
            bboxes = np.array(dic['bboxes'])
            classes = np.array(dic['classes'])
            _id = dic['_id']

            image_path = f'{self.data_dir}/images/{task}/{_id}'
            label_path = f'{self.data_dir}/labels/{task}/{Path(_id).stem}.txt'

            if image_type == DataRegister.PATH:
                shutil.copy(image, image_path)
            elif image_type == DataRegister.IMAGE:
                cv2.imwrite(image, image_path)
            else:
                raise ValueError(f'Unknown input {image_type = }')

            label = np.c_[classes, bboxes]

            np.savetxt(label_path, label, fmt='%.3f')

            f.write(image_path + '\n')

        f.close()

    @staticmethod
    def _default_convert(x):
        h, w, c = x['image'].shape

        # ref labels convert to abs labels
        x['bboxes'] = converter.CoordinateConvert.mid_xywh2top_xyxy(x['bboxes'], wh=(w, h), blow_up=True)

        return x

    def save_full_labels(self, data, confs=None, convert_func=_default_convert):
        """format of saved label txt like (class, conf, w, h, c, x1, y1, x2, y2)

        Args:
            data: accept a dict like Loader().load_total() return
            confs: model predictions
            convert_func

        """
        for i, tmp in enumerate(tqdm(data)):
            tmp_dir = f'{self.data_dir}/full_labels/{tmp["task"]}'
            os_lib.mk_dir(tmp_dir)

            if convert_func:
                tmp = convert_func(tmp)

            h, w, c = tmp['image'].shape

            bboxes = tmp['bboxes']
            classes = tmp['classes']

            labels = np.c_[
                classes,
                [1] * len(classes) if confs is None else confs[i],
                [w] * len(classes),
                [h] * len(classes),
                [c] * len(classes),
                bboxes
            ]

            np.savetxt(f'{tmp_dir}/{Path(tmp["_id"]).stem}.txt', labels, fmt='%d')


class Generator(DataGenerator):
    image_suffix = 'png'

    def gen_sets(self, image_dirs=(), save_dir='', task='',
                 id_distinguish='', id_sort=False,
                 set_names=('train', 'test'), split_ratio=(0.8, 1)):
        """

        Args:
            image_dirs: special dir or use Generator.image_dir
            save_dir: special dir or use {Generator.data_dir}/image_sets/{task}
            task:
            id_distinguish:
                the image file name where having same id must not be split to different sets.
                e.g. id_distinguish = '_', the fmt of image file name would like '{id}_{sub_id}.png'
            id_sort: if id_distinguish is set, True for sorting by sub_ids and sub_ids must be int type
            set_names: save txt name
            split_ratio:
                split ratio for each set, the shape must apply for set_names
                if id_distinguish is set, the ration is num of ids not files

        """
        image_dirs = image_dirs or [self.image_dir]

        img_list = []
        for image_dir in image_dirs:
            img_list += [i for i in Path(image_dir).glob(f'*.{self.image_suffix}')]

        if id_distinguish:
            tmp = defaultdict(list)
            for img in img_list:
                fn = img.stem
                a, b = fn.split('_', 1)
                tmp[a].append([img, b])

            if id_sort:
                for k, v in tmp:
                    # convert str to int
                    for vv in v:
                        vv[1] = int(vv[1])

                    tmp[k] = sorted(v, key=lambda x: x[1])

            ids = list(tmp.keys())
            np.random.shuffle(ids)

            i = 0
            for j, set_name in zip(split_ratio, set_names):
                i, j = int(i * len(ids)), int(j * len(ids))
                img_list = []
                for k in ids[i:j]:
                    img_list += [vv[0] for vv in tmp[k]]

                with open(f'{save_dir}/{set_name}.txt', 'w', encoding='utf8') as f:
                    for img in img_list[i:j]:
                        f.write(os.path.abspath(img) + '\n')

        else:
            np.random.shuffle(img_list)

            save_dir = save_dir or f'{self.data_dir}/image_sets/{task}'

            i = 0
            for j, set_name in zip(split_ratio, set_names):
                i, j = int(i * len(img_list)), int(j * len(img_list))
                with open(f'{save_dir}/{set_name}.txt', 'w', encoding='utf8') as f:
                    for img in img_list[i:j]:
                        f.write(os.path.abspath(img) + '\n')
