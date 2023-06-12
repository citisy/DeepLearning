from tqdm import tqdm
from utils import os_lib
import pickle
from enum import Enum
from pathlib import Path
from collections import defaultdict
import numpy as np


class DataRegister(Enum):
    place_holder = None

    MIX = 'mix'
    ALL = 'all'
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'

    PATH = 1
    IMAGE = 2


class DataLoader:
    default_set_type = [DataRegister.TRAIN, DataRegister.TEST]
    default_data_type = DataRegister.ALL
    default_image_type = DataRegister.PATH
    image_suffix = 'jpg'
    classes = []

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __call__(self, set_type=None, image_type=None, generator=True, **kwargs):
        """
        Args:
            set_type(list or DataRegister): a DataRegister type or a list of them
                Mix -> [DataRegister.place_holder]
                ALL -> DataLoader.default_set_type
                other set_type -> [set_type]
            image_type(DataRegister): `DataRegister.PATH` or `DataRegister.IMAGE`
                PATH -> a str of image abs path
                IMAGE -> a np.ndarray of image, read from cv2, as (h, w, c)
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
                for _ in tqdm(self._call(set_type, image_type, **kwargs)):
                    tmp.append(_)

                r.append(tmp)

        return r

    def _call(self, set_type, image_type, **kwargs):
        """

        Args:
            set_type(DataRegister): a DataRegister type, see also `DataLoader.__call__`
            image_type(DataRegister): `DataRegister.PATH` or `DataRegister.IMAGE`
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


class DataSaver:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.default_set_type = [DataRegister.TRAIN, DataRegister.TEST]

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
            image_type(DataRegister): `DataRegister.PATH` or `DataRegister.IMAGE`
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


class DataGenerator:
    image_suffix = 'jpg'

    def __init__(self, data_dir=None, image_dir=None, label_dir=None):
        self.data_dir = data_dir
        self.image_dir = image_dir
        self.label_dir = label_dir

    @staticmethod
    def _gen_sets(
            data, idx=None, id_distinguish='', id_sort=False, save_dir=None,
            set_names=('train', 'test'), split_ratio=(0.8, 1), **kwargs
    ):
        if id_distinguish:
            tmp = defaultdict(list)
            for i, (d, _idx) in enumerate(zip(data, idx)):
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
                        except:
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
                        f.write(data[candidate_id] + '\n')

                i = j

        else:
            np.random.shuffle(data)

            i = 0
            for j, set_name in zip(split_ratio, set_names):
                j = int(j * len(data))
                with open(f'{save_dir}/{set_name}.txt', 'w', encoding='utf8') as f:
                    f.write('\n'.join(data[i:j]))

                i = j
