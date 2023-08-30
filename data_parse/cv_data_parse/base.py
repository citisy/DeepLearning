import cv2
import pickle
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from utils import os_lib, converter, visualize
from typing import List
from numbers import Number
from .. import DataRegister


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


class ImgClsDict:
    # image file name, e.g. 'xxx.png'
    _id: str

    # if `image_type = DataRegister.PATH`,, return a string
    # if `image_type = DataRegister.ARRAY`,, return a np.ndarray from cv2.read()
    image: str or np.ndarray

    # usually an int number giving the pic class
    _class: Number


class ObjDetDict:
    # image file name, e.g. 'xxx.png'
    _id: str

    # if `image_type = DataRegister.PATH`,, return a string
    # if `image_type = DataRegister.ARRAY`,, return a np.ndarray from cv2.read()
    image: str or np.ndarray

    # a np.ndarray with shape of (n_obj, 4), 4 gives [top_left_x, top_left_y, right_down_x, right_down_y] usually
    bboxes: np.ndarray

    # a np.ndarray with shape of (n_obj, )
    classes: np.ndarray


class ImgSegDict:
    # image file name, e.g. 'xxx.png'
    _id: str

    # if `image_type = DataRegister.PATH`,, return a string
    # if `image_type = DataRegister.ARRAY`,, return a np.ndarray from cv2.read()
    image: str or np.ndarray

    # if `image_type = DataRegister.PATH`,, return a string
    # if `image_type = DataRegister.ARRAY`,, return a np.ndarray from cv2.read()
    pix_image: str or np.ndarray


def fake_func(x):
    return x


class DataLoader:
    default_set_type = [DataRegister.TRAIN, DataRegister.TEST]
    default_data_type = DataRegister.FULL
    default_image_type = DataRegister.PATH
    image_suffix = 'png'
    classes = []

    def __init__(self, data_dir, verbose=True, stdout_method=print):
        self.data_dir = data_dir
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else os_lib.FakeIo()

    def __call__(self, *args, **kwargs):
        return self.load(*args, **kwargs)

    def load(self, set_type=None, image_type=None, generator=True,
             use_multiprocess=False, n_process=5,
             **load_kwargs):
        """
        Args:
            set_type(list or DataRegister): a DataRegister type or a list of them
                FULL -> DataLoader.default_set_type
                other set_type -> [set_type]
            image_type(DataRegister): `DataRegister.PATH` or `DataRegister.ARRAY`
                PATH -> a str of image abs path
                ARRAY -> a np.ndarray of image, read from cv2, as (h, w, c)
            generator(bool):
                return a generator if True else a list
            use_multiprocess(bool):
                whether used multiprocess to load data or not
            n_process(int):
                num of process pools to execute if used multiprocess
            load_kwargs:
                see also `_call` function to get more details of load_kwargs
                see also 'gen_data' function to get more details of gen_kwargs
                see also 'get_ret' function to get more details of get_kwargs

        Returns:
            a list apply for set_type
            e.g.
                set_type=DataRegister.TRAIN, return a list of [DataRegister.TRAIN]
                set_type=[DataRegister.TRAIN, DataRegister.TEST], return a list of them
        """
        set_type = set_type or self.default_data_type
        image_type = image_type or self.default_image_type

        if set_type == DataRegister.FULL:
            set_types = self.default_set_type
        elif isinstance(set_type, list):
            set_types = set_type
        elif isinstance(set_type, DataRegister):
            set_types = [set_type]
        else:
            raise ValueError(f'Unknown input {set_type = }')

        r = []
        for set_type in set_types:
            pbar = self._call(set_type=set_type, image_type=image_type, **load_kwargs)

            if use_multiprocess:
                from multiprocessing.pool import Pool

                pool = Pool(n_process)
                pbar = pool.imap(fake_func, pbar)

            if self.verbose:
                pbar = tqdm(pbar, desc=visualize.TextVisualize.highlight_str(f'Load {set_type.value} dataset'))

            if generator:
                r.append(pbar)
            else:
                r.append(list(pbar))

        return r

    def _call(self, **gen_kwargs):
        """
        Args:
            gen_kwargs:
                see also `gen_data` function to get more details of gen_kwargs
                see also `get_ret` function to get more details of get_kwargs

        Returns:
            a generator which yield a dict of data

        Usage:
            .. code-block:: python

            def _call(self, **gen_kwargs)
                gen_func = ...
                return self.gen_data(gen_func, **gen_kwargs)
        """
        raise NotImplementedError

    def gen_data(self, gen_func, max_size=float('inf'), **get_kwargs):
        """

        Args:
            gen_func:
            max_size: num of loaded data
            **get_kwargs:
                see also `get_ret` function to get more details of get_kwargs

        Yields
            a dict of result data

        """
        i = 0
        for a in gen_func:
            if i >= max_size:
                break

            ret = self.get_ret(a, **get_kwargs)
            if not ret:
                continue

            ret = self.convert_func(ret)
            if self.filter_func(ret):
                i += 1
                yield ret

    def get_ret(self, *args, **kwargs) -> dict:
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
        self.stdout_method = stdout_method if verbose else os_lib.FakeIo()

        os_lib.mk_dir(self.data_dir)

    def __call__(self, *args, **kwargs):
        return self.save(*args, **kwargs)

    def save(self, data, set_type=DataRegister.FULL, image_type=DataRegister.PATH, **kwargs):
        """

        Args:
            data(list): a list apply for set_type
                See Also return of `DataLoader.__call__`
            set_type(list or DataRegister): a DataRegister type or a list of them
                ALL -> DataLoader.default_set_type
                other set_type -> [set_type]
            image_type(DataRegister): `DataRegister.PATH` or `DataRegister.ARRAY`
                PATH -> a str of image abs path
                IMAGE -> a np.ndarray of image, read from cv2, as (h, w, c)

        """
        if set_type == DataRegister.FULL:
            set_types = self.default_set_type
        elif isinstance(set_type, list):
            set_types = set_type
        elif isinstance(set_type, DataRegister):
            set_types = [set_type]
        else:
            raise ValueError(f'Unknown input {set_type = }')

        self.mkdirs(set_types, **kwargs)

        for i, iter_data in enumerate(data):
            if self.verbose:
                iter_data = tqdm(iter_data, desc=visualize.TextVisualize.highlight_str(f'Save {set_type.value} dataset'))

            self._call(iter_data, set_type=set_types[i], image_type=image_type, **kwargs)

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
        self.stdout_method = stdout_method if verbose else os_lib.FakeIo()

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

    def __init__(self, save_dir, pbar=True, **saver_kwargs):
        self.save_dir = save_dir
        saver_kwargs.setdefault('verbose', not pbar)
        self.saver = os_lib.Saver(**saver_kwargs)
        self.pbar = pbar
        os_lib.mk_dir(save_dir)

    def __call__(self, *iter_data, max_vis_num=None, return_image=False, **visual_kwargs):
        """

        Args:
            *iter_data (List[dict]):
                each dict must have the of '_id', 'image' and at lease
                    - _id (str): the name to save the image
                    - image (np.ndarray): must have the same shape
                    - pix_image:
                    - bboxes:
                    - classes:
                    - confs:
                    - colors:

            **visual_kwargs:
                cls_alias:
                return_image:

        Returns:

        """
        pbar = zip(*iter_data)
        if self.pbar:
            pbar = tqdm(pbar, desc='visual')

        vis_num = 0
        max_vis_num = max_vis_num or float('inf')
        cache_image = []
        for rets in pbar:
            if vis_num >= max_vis_num:
                return

            images = []
            _id = 'tmp.png'
            for r in rets:
                images.append(self.visual_one_image(r, **visual_kwargs))

                if 'pix_image' in r:
                    images.append(self.visual_one_image({'image': r['pix_image']}, **visual_kwargs))

                if '_id' in r:
                    _id = r['_id']

            image = self.concat_images(images)
            self.saver.save_img(image, f'{self.save_dir}/{_id}')
            vis_num += 1
            if return_image:
                cache_image.append(image)

        return cache_image

    def visual_one_image(self, r, **visual_kwargs):
        image = r['image']

        if 'bboxes' in r and r['bboxes'] is not None:
            bboxes = r['bboxes']

            if 'classes' in r and r['classes'] is not None:
                classes = r['classes']

                if 'colors' in r:
                    colors = r['colors']
                else:
                    colors = [visualize.get_color_array(int(cls)) for cls in classes]

                if 'cls_alias' in visual_kwargs and visual_kwargs['cls_alias']:
                    cls_alias = visual_kwargs['cls_alias']
                    classes = [cls_alias[_] for _ in classes]

                if 'confs' in r:
                    classes = [f'{cls} {conf:.6f}' for cls, conf in zip(classes, r['confs'])]

                image = visualize.ImageVisualize.label_box(image, bboxes, classes, colors=colors)

            else:
                if 'colors' in r:
                    colors = r['colors']
                else:
                    colors = [visualize.cmap['Black']['array'] for _ in bboxes]

                image = visualize.ImageVisualize.box(image, bboxes, colors=colors)

        return image

    def concat_images(self, images):
        n = len(images)
        if n < 4:
            return np.concatenate(images, 1)

        n_col = int(np.ceil(np.sqrt(n)))
        n_row = int(np.ceil(n / n_col))
        images += [np.zeros_like(images[0])] * (n_col * n_row - n)
        images = [np.concatenate(images[i: i + n_col], 1) for i in range(0, len(images), n_col)]
        return np.concatenate(images, 0)
