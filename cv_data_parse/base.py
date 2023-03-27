from tqdm import tqdm
from utils import os_lib
import pickle
from enum import Enum


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
    default_load_type = [DataRegister.TRAIN, DataRegister.TEST]
    default_data_type = DataRegister.ALL
    default_image_type = DataRegister.PATH
    image_suffix = 'jpg'
    classes = []

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __call__(self, data_type=None, image_type=None, generator=True, **kwargs):
        """
        Args:
            data_type: Register.ALL, Register.TRAIN, Register.TEST or list of them
            image_type: Register.FILE or Register.IMAGE
            generator: would be returned that `True` for a generator or `False` for a list
        """
        data_type = data_type or self.default_data_type
        image_type = image_type or self.default_image_type

        if data_type == DataRegister.MIX:
            load_types = [DataRegister.place_holder]
        elif data_type == DataRegister.ALL:
            load_types = self.default_load_type
        elif isinstance(data_type, list):
            load_types = [_ for _ in data_type]
        elif isinstance(data_type, DataRegister):
            load_types = [data_type]
        else:
            raise ValueError(f'Unknown input {data_type = }')

        r = []
        for load_type in load_types:
            tmp = []
            if generator:
                r.append(self._call(load_type, image_type, **kwargs))

            else:
                for _ in tqdm(self._call(load_type, image_type, **kwargs)):
                    tmp.append(_)

                r.append(tmp)

        return r

    def _call(self, load_type, image_type, **kwargs):
        raise NotImplementedError

    def load_cache(self, save_name):
        with open(f'{self.data_dir}/cache/{save_name}.pkl', 'rb') as f:
            data = pickle.load(f)

        return data


class DataSaver:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.default_load_type = [DataRegister.TRAIN, DataRegister.TEST]

        os_lib.mk_dir(self.data_dir)

    def __call__(self, data, data_type=DataRegister.ALL, image_type=DataRegister.PATH, **kwargs):
        if data_type == DataRegister.MIX:
            load_types = [DataRegister.place_holder]
        elif data_type == DataRegister.ALL:
            load_types = self.default_load_type
        elif isinstance(data_type, list):
            load_types = [_ for _ in data_type]
        elif isinstance(data_type, DataRegister):
            load_types = [data_type]
        else:
            raise ValueError(f'Unknown input {data_type = }')

        for i, iter_data in enumerate(data):
            self._call(iter_data, load_types[i], image_type, **kwargs)

    def _call(self, iter_data, load_type, image_type, **kwargs):
        raise ValueError

    def save_cache(self, data, save_name):
        save_dir = f'{self.data_dir}/cache'
        os_lib.mk_dir(save_dir)

        with open(f'{save_dir}/{save_name}.pkl', 'wb') as f:
            pickle.dump(data, f)


class DataGenerator:
    image_suffix = 'jpg'

    def __init__(self, data_dir=None, image_dir=None):
        self.data_dir = data_dir
        self.image_dir = image_dir
