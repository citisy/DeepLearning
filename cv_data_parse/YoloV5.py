import cv2
import os
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Iterable
from utils import os_lib, converter, visualize
from cv_data_parse.base import DataRegister, DataLoader, DataSaver, DataGenerator, get_image, save_image, DataVisualizer


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
            data = loader(set_type=DataRegister.ALL, generator=True, image_type=DataRegister.ARRAY)
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

    def convert_func(self, x):
        if isinstance(x['image'], np.ndarray):
            h, w, c = x['image'].shape

            # ref labels convert to abs labels
            x['bboxes'] = converter.CoordinateConvert.mid_xywh2top_xyxy(x['bboxes'], wh=(w, h), blow_up=True)

        return x

    def _call(self, set_type, image_type, task='', set_task='', **kwargs):
        """See Also `cv_data_parse.base.DataLoader._call`

        Args:
            set_type:
            image_type:
            task(str): one of dir name in `images` dir
            set_task(str): one of dir name in `image_sets` dir

        Returns:
            a dict had keys of
                _id: image file name
                image: see also image_type
                bboxes: a np.ndarray with shape of (-1, 4), 4 means [center_x, center_y, w, h] after norm
                classes: a list
        """

        if set_type == DataRegister.place_holder:
            return self.load_total(image_type, task, **kwargs)
        else:
            return self.load_set(set_type, image_type, set_task, **kwargs)

    def load_total(self, image_type, task='', convert_func=None, **kwargs):
        for img_fp in Path(f'{self.data_dir}/images/{task}').glob(f'*.{self.image_suffix}'):
            image_path = os.path.abspath(img_fp)
            img_fp = Path(image_path)
            image = get_image(image_path, image_type)
            labels = np.genfromtxt(image_path.replace('images', 'labels').replace(f'.{self.image_suffix}', '.txt')).reshape((-1, 5))

            # (center x, center y, box w, box h) after norm
            # e.g. norm box w = real box w / real image w
            bboxes = labels[:, 1:]
            classes = labels[:, 0].astype(int)

            ret = dict(
                _id=img_fp.name,
                image_dir=str(img_fp.parent),
                image=image,
                bboxes=bboxes,
                classes=classes,
            )

            if convert_func:
                ret = convert_func(ret)

            yield ret

    def load_set(self, set_type, image_type, set_task='', convert_func=None, sub_dir='image_sets', **kwargs):
        with open(f'{self.data_dir}/{sub_dir}/{set_task}/{set_type.value}.txt', 'r', encoding='utf8') as f:
            for line in f.read().strip().split('\n'):
                image_path = os.path.abspath(line)
                img_fp = Path(image_path)
                image = get_image(image_path, image_type)
                labels = np.genfromtxt(image_path.replace('images', 'labels').replace(f'.{self.image_suffix}', '.txt')).reshape((-1, 5))

                # (center x, center y, box w, box h) after norm
                # e.g. norm box w = real box w / real image w
                bboxes = labels[:, 1:]
                classes = labels[:, 0]

                ret = dict(
                    _id=img_fp.name,
                    image_dir=str(img_fp.parent),
                    image=image,
                    bboxes=bboxes,  # (-1, 4)
                    classes=classes,
                )

                if convert_func:
                    ret = convert_func(ret)

                yield ret

    def load_full_labels(self, task='', sub_dir='full_labels', convert_func=None):
        """format of saved label txt like (class, x1, y1, x2, y2, conf, w, h)
        only return labels but no images. can use _id to load images"""
        for fp in Path(f'{self.data_dir}/{sub_dir}/{task}').glob('*.txt'):
            img_fp = Path(fp.name.replace('.txt', '.png'))

            # (class, x1, y1, x2, y2, conf, w, h)
            labels = np.genfromtxt(fp)
            if len(labels.shape) == 1:
                labels = labels[None, :]

            ret = dict(
                _id=img_fp.name,
                image_dir=str(img_fp.parent),
                task=task,
                classes=labels[:, 0],
                bboxes=labels[:, 1:5],
            )

            if labels.shape[-1] > 5:
                ret.update(
                    confs=labels[:, 5],
                    image_shape=labels[:, 6:8],  # (w, h)
                )

            if convert_func:
                ret = convert_func(ret)

            yield ret


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
            data = loader(set_type=DataRegister.TRAIN)

            # save as yolov5 type
            from cv_data_parse.YoloV5 import Saver
            saver = Saver('data/Yolov5')
            saver(data, set_type=DataRegister.TRAIN)

    """

    def __call__(self, data, set_type=DataRegister.ALL, image_type=DataRegister.PATH, **kwargs):
        task = kwargs.get('task', '')
        set_task = kwargs.get('set_task', '')

        os_lib.mk_dir(f'{self.data_dir}/image_sets/{set_task}')
        os_lib.mk_dir(f'{self.data_dir}/images/{task}')
        os_lib.mk_dir(f'{self.data_dir}/labels/{task}')

        super().__call__(data, set_type, image_type, **kwargs)

    def _call(self, iter_data, set_type, image_type, convert_func=None, **kwargs):
        """

        Args:
            iter_data (Iterable[dict]):
                list of dict which has the key of _id, image, bboxes, classes
            set_type:
            image_type:
            convert_func:
                as inputs, the bboxes of standard dataset type is abs top xyxy usually,
                but as outputs, the bboxes of office yolov5 type is ref center xywh,
                so can be use a convert function to change the bboxes
            **kwargs:

        Returns:

        """
        task = kwargs.get('task', '')
        set_task = kwargs.get('set_task', '')

        if set_type == DataRegister.place_holder:
            f = os_lib.FakeIo()
        else:
            f = open(f'{self.data_dir}/image_sets/{set_task}/{set_type.value}.txt', 'w', encoding='utf8')

        for dic in tqdm(iter_data):
            if image_type == DataRegister.ARRAY and convert_func:
                dic = convert_func(dic)

            image = dic['image']
            bboxes = np.array(dic['bboxes'])
            classes = np.array(dic['classes'])
            _id = dic['_id']

            image_path = f'{self.data_dir}/images/{task}/{_id}'
            label_path = f'{self.data_dir}/labels/{task}/{Path(_id).stem}.txt'
            save_image(image, image_path, image_type)
            label = np.c_[classes, bboxes]

            np.savetxt(label_path, label, fmt='%.6f')

            f.write(image_path + '\n')

        f.close()

    def save_full_labels(self, iter_data, sub_dir='full_labels', task='', convert_func=None):
        """format of saved label txt like (class, x1, y1, x2, y2, conf, w, h)

        Args:
            iter_data (Iterable[dict]):
                accept a dict like Loader().load_total() return
                list of dict which has the key of _id, image(non-essential), bboxes, classes, confs
            convert_func

        """
        save_dir = f'{self.data_dir}/{sub_dir}/{task}'
        os_lib.mk_dir(save_dir)

        for i, dic in enumerate(tqdm(iter_data)):
            if convert_func:
                dic = convert_func(dic)

            if 'image' in dic:
                h, w, c = dic['image'].shape
            else:
                h, w = -1, -1

            bboxes = dic['bboxes']
            classes = dic['classes']

            labels = np.c_[
                classes,
                bboxes,
                dic['confs'] if 'confs' in dic else [1] * len(classes),
                [w] * len(classes),
                [h] * len(classes),
            ]

            np.savetxt(f'{save_dir}/{Path(dic["_id"]).stem}.txt', labels, fmt='%.6f')


class Generator(DataGenerator):
    image_suffix = 'png'
    label_suffix = 'txt'

    def gen_sets(self, label_dirs=(), image_dirs=(), save_dir='', set_task='',
                 id_distinguish='', id_sort=False,
                 set_names=('train', 'val'), split_ratio=(0.8, 1), **kwargs):
        """

        Args:
            label_dirs: special dir or if image_dirs is set, use image_dirs, else use Generator.label_dir
            image_dirs: special dir or use Generator.image_dir, if label_dirs is set, ignore this param
            save_dir: special dir or use {Generator.data_dir}/image_sets/{task}
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
            └── image_sets
                └── [set_task]
                      ├── train.txt  # per image file path per line
                      ├── test.txt   # would like to be empty or same to val.txt
                      └── val.txt

        Usage:
            .. code-block:: python

                from cv_data_parse.PaddleOcr import Generator

                # single data dir
                gen = Generator(
                    data_dir='data/yolov5',
                    label_dir='data/yolov5/labels/1',
                )
                gen.gen_sets(set_task='1')

                # multi data dir
                gen = Generator()
                gen.gen_sets(
                    label_files=('data/yolov5_0/labels', 'data/yolov5_1/labels'),
                    save_dir='data/yolov5_0_1/image_sets'
                )
        """
        save_dir = save_dir or f'{self.data_dir}/image_sets/{set_task}'
        os_lib.mk_dir(save_dir)

        if not label_dirs and not image_dirs and self.label_dir:
            label_dirs = [self.label_dir]

        image_dirs = image_dirs or [self.image_dir]

        data = []
        idx = []

        if label_dirs:
            for label_dir in label_dirs:
                tmp = list(Path(label_dir).glob(f'*.{self.label_suffix}'))

                if id_distinguish:
                    idx += [i.stem for i in tmp]

                tmp = [os.path.abspath(str(i).replace(self.label_suffix, self.image_suffix)) for i in tmp]
                data += tmp

        else:
            for image_dir in image_dirs:
                tmp = list(Path(image_dir).glob(f'*.{self.image_suffix}'))

                if id_distinguish:
                    idx += [i.stem for i in tmp]

                tmp = [os.path.abspath(i) for i in tmp]
                data += tmp

        self._gen_sets(data, idx, id_distinguish, id_sort, save_dir, set_names, split_ratio)


class Visualizer(DataVisualizer):
    def visual_one_image(self, r, cls_alias=None):
        image = r['image']
        bboxes = r['bboxes']
        classes = r['classes']
        colors = [visualize.get_color_array(int(cls)) for cls in classes]

        if cls_alias:
            classes = [cls_alias[_] for _ in classes]

        if 'confs' in r:
            classes = [f'{cls} {conf:.6f}' for cls, conf in zip(classes, r['confs'])]

        image = visualize.ImageVisualize.label_box(image, bboxes, classes, colors=colors, line_thickness=2)

        return image
