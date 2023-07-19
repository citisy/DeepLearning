import cv2
import pickle
import shutil
import numpy as np
from tqdm import tqdm
from enum import Enum
from pathlib import Path
from collections import defaultdict
from utils import os_lib, converter, visualize
from typing import List


class DataRegister(Enum):
    place_holder = None

    MIX = 'mix'
    ALL = 'all'
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'
    DEV = 'dev'
    TRAIN_VAL = 'trainval'

    PATH = 1
    ARRAY = 2
    BASE64 = 3


def get_image(obj: str, image_type):
    if image_type == DataRegister.PATH:
        image = obj
    elif image_type == DataRegister.ARRAY:
        image = cv2.imread(obj)
    elif image_type == DataRegister.BASE64:
        image = cv2.imread(obj)
        image = converter.DataConvert.image_to_base64(image)
    else:
        raise ValueError(f'Unknown input {image_type = }')

    return image


def save_image(obj, save_path, image_type):
    if image_type == DataRegister.PATH:
        shutil.copy(obj, save_path)
    elif image_type == DataRegister.ARRAY:
        cv2.imwrite(save_path, obj)
    elif image_type == DataRegister.BASE64:
        obj = converter.DataConvert.base64_to_image(obj)
        cv2.imwrite(save_path, obj)
    else:
        raise ValueError(f'Unknown input {image_type = }')


class DataLoader:
    default_set_type = [DataRegister.TRAIN, DataRegister.TEST]
    default_data_type = DataRegister.ALL
    default_image_type = DataRegister.PATH
    image_suffix = 'jpg'
    classes = []

    def __init__(self, data_dir, verbose=True, stdout_method=print):
        self.data_dir = data_dir
        self.verbose = verbose
        self.stdout_method = stdout_method

    def __call__(self, set_type=None, image_type=None, generator=True, **kwargs):
        """
        Args:
            set_type(list or DataRegister): a DataRegister type or a list of them
                Mix -> [DataRegister.place_holder]
                ALL -> DataLoader.default_set_type
                other set_type -> [set_type]
            image_type(DataRegister): `DataRegister.PATH` or `DataRegister.ARRAY`
                PATH -> a str of image abs path
                ARRAY -> a np.ndarray of image, read from cv2, as (h, w, c)
            generator(bool):
                return a generator if True else a list

        Returns:
            a list apply for set_type
            e.g.
                set_type=DataRegister.TRAIN, return a list of [DataRegister.TRAIN]
                set_type=[DataRegister.TRAIN, DataRegister.TEST], return a list of them
        """
        set_type = set_type or self.default_data_type
        image_type = image_type or self.default_image_type

        if set_type == DataRegister.MIX:
            set_types = [DataRegister.place_holder]
        elif set_type == DataRegister.ALL:
            set_types = self.default_set_type
        elif isinstance(set_type, list):
            set_types = set_type
        elif isinstance(set_type, DataRegister):
            set_types = [set_type]
        else:
            raise ValueError(f'Unknown input {set_type = }')

        r = []
        for set_type in set_types:
            tmp = []
            if generator:
                r.append(self._call(set_type, image_type, **kwargs))

            else:
                for _ in tqdm(self._call(set_type, image_type, **kwargs), desc=f'Load {set_type.value} dataset'):
                    tmp.append(_)

                r.append(tmp)

        return r

    def _call(self, set_type, image_type, **kwargs):
        """

        Args:
            set_type(DataRegister): a DataRegister type, see also `DataLoader.__call__`
            image_type(DataRegister): `DataRegister.PATH` or `DataRegister.ARRAY`
                PATH -> a str of image abs path
                IMAGE -> a np.ndarray of image, read from cv2, as (h, w, c)

        Returns:
            yield a dict of loaded data

        """
        raise NotImplementedError

    def load_cache(self, save_name):
        with open(f'{self.data_dir}/cache/{save_name}.pkl', 'rb') as f:
            data = pickle.load(f)

        return data

    def convert_func(self, ret):
        return ret

    def filter_func(self, ret):
        return True


class DataSaver:
    def __init__(self, data_dir, verbose=True, stdout_method=print):
        self.data_dir = data_dir
        self.default_set_type = [DataRegister.TRAIN, DataRegister.TEST]
        self.verbose = verbose
        self.stdout_method = stdout_method

        os_lib.mk_dir(self.data_dir)

    def __call__(self, data, set_type=DataRegister.ALL, image_type=DataRegister.PATH, **kwargs):
        """

        Args:
            data(list): a list apply for set_type
                See Also return of `DataLoader.__call__`
            set_type(list or DataRegister): a DataRegister type or a list of them
                Mix -> [DataRegister.place_holder]
                ALL -> DataLoader.default_set_type
                other set_type -> [set_type]
            image_type(DataRegister): `DataRegister.PATH` or `DataRegister.ARRAY`
                PATH -> a str of image abs path
                IMAGE -> a np.ndarray of image, read from cv2, as (h, w, c)

        """
        if set_type == DataRegister.MIX:
            set_types = [DataRegister.place_holder]
        elif set_type == DataRegister.ALL:
            set_types = self.default_set_type
        elif isinstance(set_type, list):
            set_types = set_type
        elif isinstance(set_type, DataRegister):
            set_types = [set_type]
        else:
            raise ValueError(f'Unknown input {set_type = }')

        self.mkdirs(set_types, **kwargs)

        for i, iter_data in enumerate(data):
            self._call(iter_data, set_types[i], image_type, **kwargs)

    def mkdirs(self, set_types, **kwargs):
        pass

    def _call(self, iter_data, set_type, image_type, **kwargs):
        raise ValueError

    def save_cache(self, data, save_name):
        save_dir = f'{self.data_dir}/cache'
        os_lib.mk_dir(save_dir)

        with open(f'{save_dir}/{save_name}.pkl', 'wb') as f:
            pickle.dump(data, f)

    def convert_func(self, ret):
        return ret

    def filter_func(self, ret):
        return True


class DatasetGenerator:
    """generate datasets for training, testing and valuating"""
    image_suffix = 'jpg'

    def __init__(self, data_dir=None, image_dir=None, label_dir=None, verbose=True, stdout_method=print):
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.verbose = verbose
        self.stdout_method = stdout_method

    def gen_sets(self, **kwargs):
        """please implement this function"""
        return self._gen_sets(**kwargs)

    @staticmethod
    def _gen_sets(
            iter_data, idx=None, id_distinguish='', id_sort=False, save_dir=None,
            set_names=('train', 'test'), split_ratio=(0.8, 1.), **kwargs
    ):
        """

        Args:
            iter_data (List[str]):
            idx (List[str]): be used for sorting
            id_distinguish:
                the image file name where having same id must not be split to different sets.
                e.g. id_distinguish = '_', the fmt of image file name would like '{id}_{sub_id}.png'
            id_sort: if id_distinguish is set, True for sorting by sub_ids and sub_ids must be int type
            save_dir: save_dir
            set_names: save txt name
            split_ratio:
                split ratio for each set, the shape must apply for set_names
                if id_distinguish is set, the ration is num of ids not files
            **kwargs:

        Returns:

        """
        if id_distinguish:
            tmp = defaultdict(list)
            for i, (d, _idx) in enumerate(zip(iter_data, idx)):
                stem = Path(_idx).stem
                _ = stem.split(id_distinguish)
                a, b = id_distinguish.join(_[:-1]), _[-1]
                tmp[a].append([i, b])

            if id_sort:
                for k, v in tmp.items():
                    # convert str to int
                    for vv in v:
                        try:
                            vv[1] = int(vv[1])
                        except ValueError:
                            pass

                    tmp[k] = sorted(v, key=lambda x: x[1])

            ids = list(tmp.keys())
            np.random.shuffle(ids)

            i = 0
            for j, set_name in zip(split_ratio, set_names):
                j = int(j * len(ids))
                candidate_ids = []
                for k in ids[i:j]:
                    candidate_ids += [vv[0] for vv in tmp[k]]

                if not id_sort:
                    np.random.shuffle(candidate_ids)

                with open(f'{save_dir}/{set_name}.txt', 'w', encoding='utf8') as f:
                    for candidate_id in candidate_ids:
                        f.write(iter_data[candidate_id] + '\n')

                i = j

        else:
            np.random.shuffle(iter_data)

            i = 0
            for j, set_name in zip(split_ratio, set_names):
                j = int(j * len(iter_data))
                with open(f'{save_dir}/{set_name}.txt', 'w', encoding='utf8') as f:
                    f.write('\n'.join(iter_data[i:j]))

                i = j

    def filter_func(self, x):
        return True


class DataVisualizer:
    """
    Usage:
        .. code-block:: python

            def visual_one_image(r, **visual_kwargs):
                image = r['image']

                if 'bboxes' in r:
                    bboxes = r['bboxes']
                    classes = r['classes']
                    colors = [visualize.get_color_array(int(cls)) for cls in classes]

                    image = visualize.ImageVisualize.block(image, bboxes, colors=colors)

                return image

            visualizer = DataVisualizer('visuals', verbose=False)

            # use special visual method
            visualizer.visual_one_image = visual_one_image

            visualizer([{'images': images, 'bboxes': bboxes, 'classes': classes}])
    """
    def __init__(self, save_dir, verbose=True, stdout_method=print, **saver_kwargs):
        self.save_dir = save_dir
        self.saver = os_lib.Saver(verbose=verbose, stdout_method=stdout_method, **saver_kwargs)
        self.verbose = verbose
        self.stdout_method = stdout_method
        os_lib.mk_dir(save_dir)

    def __call__(self, *iter_data, **visual_kwargs):
        """

        Args:
            *iter_data (List[dict]):
                each dict must have the of 'image' and '_id' at lease
                    - image (np.ndarray): must have the same shape
                    - _id (str): the name to save the image
            **visual_kwargs:

        Returns:

        """
        for rets in tqdm(zip(*iter_data), desc='visual'):
            images = []
            _id = ''
            for r in rets:
                images.append(self.visual_one_image(r, **visual_kwargs))
                if 'pix_image' in r:
                    images.append(self.visual_one_image({'image': r['pix_image']}, **visual_kwargs))
                _id = r['_id']

            image = self.concat_images(images)
            self.saver.save_img(image, f'{self.save_dir}/{_id}')

    def visual_one_image(self, r, **visual_kwargs):
        image = r['image']

        if 'bboxes' in r and r['bboxes'] is not None:
            bboxes = r['bboxes']
            classes = r['classes']
            colors = [visualize.get_color_array(int(cls)) for cls in classes]

            if 'cls_alias' in visual_kwargs:
                cls_alias = visual_kwargs['cls_alias']
                classes = [cls_alias[_] for _ in classes]

            if 'confs' in r:
                classes = [f'{cls} {conf:.6f}' for cls, conf in zip(classes, r['confs'])]

            image = visualize.ImageVisualize.label_box(image, bboxes, classes, colors=colors)

        return image

    def concat_images(self, images):
        n = len(images)
        if n < 4:
            return np.concatenate(images, 1)

        n_row = int(np.ceil(np.sqrt(n)))
        images += [np.zeros_like(images[0])] * (n_row * n_row - n)

        images = [np.concatenate(images[i: i + n_row], 1) for i in range(0, len(images), n_row)]

        return np.concatenate(images, 0)

