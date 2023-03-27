from tqdm import tqdm
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

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __call__(self, data_type=DataRegister.ALL, generator=True, **kwargs):
        """
        Args:
            data_type: Register.ALL, Register.TRAIN, Register.TEST or list of them
            generator: would be returned that `True` for a generator or `False` for a list
        """
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
                r.append(self._call(load_type, **kwargs))

            else:
                for _ in tqdm(self._call(load_type, **kwargs)):
                    tmp.append(_)

                r.append(tmp)

        return r

    def _call(self, load_type, **kwargs):
        raise NotImplementedError
