import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Iterable
from utils import os_lib
from .base import DataRegister, DataLoader, DataSaver, DatasetGenerator, get_image, save_image, DataVisualizer


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
            from data_parse.cv_data_parse.YoloV5 import DataRegister, Loader, DataVisualizer
            from utils import converter
            from pathlib import Path

            def convert_func(ret):
                if isinstance(ret['image'], np.ndarray):
                    h, w, c = ret['image'].shape
                    ret['bboxes'] = cv_utils.CoordinateConvert.mid_xywh2top_xyxy(ret['bboxes'], wh=(w, h), blow_up=True)

                return ret

            def filter_func(ret):
                x = Path(ret['image'])
                if x.stem in filter_list:
                    image = os_lib.loader.load_img(str(x))
                    # ret['bboxes'] = cv_utils.CoordinateConvert.mid_xywh2top_xyxy(ret['bboxes'], wh=(image.shape[1], image.shape[0]), blow_up=True)
                    checkout_visualizer([ret], cls_alias=cls_alias)
                    return False

                return True

            loader = Loader('data/Yolov5Data')

            # as standard datasets, the bboxes type is abs top xyxy usually,
            # but as yolov5 official, the bboxes ref center xywh,
            # so can be use a convert function to change the bboxes
            loader.on_end_convert = convert_func

            # filter picture which is in the filter list, and checkout them
            filter_list = list()
            cls_alias = dict()
            checkout_visualizer = DataVisualizer('data/Yolov5Data/visuals/filter_samples', verbose=False)
            loader.on_end_filter = filter_func

            data = loader(set_type=DataRegister.FULL, generator=True, image_type=DataRegister.PATH)

            # visual train dataset
            DataVisualizer('data/Yolov5Data/visuals', verbose=False)(data[0])

    """

    image_suffix = 'png'

    def _call(self, set_type=DataRegister.TRAIN, **gen_kwargs):
        """

        Args:
            set_type:
            gen_kwargs:
                see also `load_full` and `load_set` functions to get more details
                see also `gen_data` function to get more details of gen_kwargs
                see also `get_ret` function to get more details of get_kwargs

        Returns:
            a generator of retuning the result dict
            see also `get_ret` function to get more details of result dict

        """

        if set_type == DataRegister.MIX:
            return self.load_mix(**gen_kwargs)
        else:
            return self.load_set(set_type=set_type, **gen_kwargs)

    def load_mix(self, task='', **gen_kwargs):
        """

        Args:
            task(str): one of dir name in `images` dir
            **gen_kwargs:
                see also `gen_data` function to get more details of gen_kwargs
                see also `get_ret` function to get more details of get_kwargs

        Returns:

        """
        gen_func = Path(f'{self.data_dir}/images/{task}').glob(f'*.{self.image_suffix}')
        return self.gen_data(gen_func, **gen_kwargs)

    def load_set(self, set_type=DataRegister.TRAIN, set_task='', sub_dir='image_sets', **gen_kwargs):
        """

        Args:
            set_type:
            set_task(str): one of dir name in `image_sets` dir
            sub_dir:
            **gen_kwargs:
                see also `gen_data` function to get more details of gen_kwargs
                see also `get_ret` function to get more details of get_kwargs

        Returns:

        """
        with open(f'{self.data_dir}/{sub_dir}/{set_task}/{set_type.value}.txt', 'r', encoding='utf8') as f:
            gen_func = f.read().strip().split('\n')
        return self.gen_data(gen_func, **gen_kwargs)

    def get_ret(self, fp, image_type=DataRegister.PATH, **kwargs) -> dict:
        """

        Args:
            fp:
            image_type:

        Returns:
            a dict having keys of
                _id: image file name
                image: see also image_type
                bboxes: a np.ndarray with shape of (-1, 4), 4 means [center_x, center_y, w, h] after norm
                classes: a list
        """
        image_path = os.path.abspath(fp)
        img_fp = Path(image_path)
        image = get_image(image_path, image_type)
        label_path = image_path.replace('images', 'labels').replace(f'.{self.image_suffix}', '.txt')

        if not os.path.exists(label_path):
            self.stdout_method(f'{label_path} not exist!')
            return {}

        labels = np.genfromtxt(label_path).reshape((-1, 5))

        # (center x, center y, box w, box h) after norm
        # e.g. norm box w = real box w / real image w
        bboxes = labels[:, 1:]
        classes = labels[:, 0].astype(int)

        return dict(
            _id=img_fp.name,
            image_dir=str(img_fp.parent),
            image=image,
            bboxes=bboxes,
            classes=classes,
        )


class LoaderFull(DataLoader):
    image_suffix = 'png'

    def _call(self, task='', sub_dir='full_labels', **gen_kwargs):
        """format of saved label txt like (class, x1, y1, x2, y2, conf, w, h)
        only return labels but no images. can use _id to load images"""
        gen_func = Path(f'{self.data_dir}/{sub_dir}/{task}').glob('*.txt')
        return self.gen_data(gen_func, task=task, sub_dir=sub_dir, **gen_kwargs)

    def get_ret(self, fp, task='', sub_dir='', **kwargs) -> dict:
        img_fp = Path(str(fp).replace('.txt', '.png').replace(sub_dir, 'images'))

        # (class, x1, y1, x2, y2, conf, w, h)
        labels = np.genfromtxt(fp)
        if not labels.size:
            labels = labels.reshape((-1, 8))

        if len(labels.shape) == 1:
            labels = labels[None, :]

        ret = dict(
            _id=img_fp.name,
            task=task,
            classes=labels[:, 0],
            bboxes=labels[:, 1:5],
        )

        if labels.shape[-1] > 5:
            ret.update(
                confs=labels[:, 5],
                image_shape=labels[:, 6:8],  # (w, h)
            )

        return ret


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

            def convert_func(ret):
                if isinstance(ret['image'], np.ndarray):
                    h, w, c = ret['image'].shape
                    ret['bboxes'] = cv_utils.CoordinateConvert.top_xyxy2mid_xywh(ret['bboxes'], wh=(w, h), blow_up=False)

                return ret

            # convert voc to yolov5
            # load data from voc
            from data_parse.cv_data_parse.Voc import Loader, DataRegister
            loader = Loader('data/VOC2012')
            data = loader(set_type=DataRegister.TRAIN)

            # save as yolov5 type
            from data_parse.cv_data_parse.YoloV5 import Saver
            saver = Saver('data/Yolov5')

            # as inputs, the bboxes of standard dataset type is abs top xyxy usually,
            # but as outputs, the bboxes of official yolov5 type is ref center xywh,
            # so can be use a convert function to change the bboxes
            saver.convert_func = convert_func

            saver(data, set_type=DataRegister.TRAIN)

    """

    def mkdirs(self, task='', set_task='', **kwargs):
        os_lib.mk_dir(f'{self.data_dir}/image_sets/{set_task}')
        os_lib.mk_dir(f'{self.data_dir}/images/{task}')
        os_lib.mk_dir(f'{self.data_dir}/labels/{task}')

    def _call(self, iter_data, set_type=DataRegister.TRAIN, set_task='', **gen_kwargs):
        """

        Args:
            iter_data (Iterable[dict]):
                list of dict which has the key of _id, image, bboxes, classes
            set_type:
            image_type:
            **kwargs:

        """
        if set_type == DataRegister.MIX:
            f = os_lib.FakeIo()
        else:
            f = open(f'{self.data_dir}/image_sets/{set_task}/{set_type.value}.txt', 'w', encoding='utf8')

        self.gen_data(iter_data, f=f, **gen_kwargs)
        f.close()

    def parse_ret(self, ret, image_type=DataRegister.PATH, f=None, task='', **get_kwargs):
        image = ret['image']
        bboxes = np.array(ret['bboxes'])
        classes = np.array(ret['classes'])
        _id = ret['_id']

        image_path = f'{self.data_dir}/images/{task}/{_id}'
        label_path = f'{self.data_dir}/labels/{task}/{Path(_id).stem}.txt'
        save_image(image, image_path, image_type)
        label = np.c_[classes, bboxes]

        np.savetxt(label_path, label, fmt='%.6f')
        f.write(image_path + '\n')


class SaverFull(DataSaver):
    def mkdirs(self, sub_dir='full_labels', task='', **kwargs):
        os_lib.mk_dir(f'{self.data_dir}/{sub_dir}/{task}')

    def _call(self, iter_data, **gen_kwargs):
        self.gen_data(iter_data, **gen_kwargs)

    def parse_ret(self, ret, sub_dir='full_labels', task='', **get_kwargs):
        """format of saved label txt like (class, x1, y1, x2, y2, conf, w, h)

        Args:
            iter_data (Iterable[dict]):
                accept a dict like Loader().load_total() return
                list of dict which has the key of _id, image(non-essential), bboxes, classes, confs
            sub_dir: labels dir
            task: image dir
        """
        if 'image' in ret:
            h, w, c = ret['image'].shape
        else:
            h, w = -1, -1

        bboxes = ret['bboxes']
        classes = ret['classes']

        labels = np.c_[
            classes,
            bboxes,
            ret['confs'] if 'confs' in ret else [1] * len(classes),
            [w] * len(classes),
            [h] * len(classes),
        ]

        np.savetxt(f'{self.data_dir}/{sub_dir}/{task}/{Path(ret["_id"]).stem}.txt', labels, fmt='%.6f')


class Generator(DatasetGenerator):
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

                from data_parse.cv_data_parse.YoloV5 import Generator

                # single data dir
                data_dir = 'data/yolov5'
                gen = Generator(
                    data_dir=data_dir,
                    label_dir=f'{data_dir}/labels/1',
                )
                gen.gen_sets(set_task='1')

                # multi data dir
                gen = Generator()
                gen.gen_sets(
                    label_dirs=('data/yolov5_0/labels', 'data/yolov5_1/labels'),
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
                tmp = [x for x in tmp if self.filter_func(x)]

                if id_distinguish:
                    idx += [i.stem for i in tmp]

                tmp = [os.path.abspath(str(i).replace(self.label_suffix, self.image_suffix)) for i in tmp]
                data += tmp

        else:
            for image_dir in image_dirs:
                tmp = list(Path(image_dir).glob(f'*.{self.image_suffix}'))
                tmp = [x for x in tmp if self.filter_func(x)]

                if id_distinguish:
                    idx += [i.stem for i in tmp]

                tmp = [os.path.abspath(i) for i in tmp]
                data += tmp

        self._gen_sets(data, idx, id_distinguish, id_sort, save_dir, set_names, split_ratio)
