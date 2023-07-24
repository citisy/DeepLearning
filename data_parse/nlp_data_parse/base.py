from tqdm import tqdm
from .. import DataRegister


class DataLoader:
    default_set_type = [DataRegister.TRAIN, DataRegister.TEST]

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def __call__(self, set_type=DataRegister.FULL, generator=True, **kwargs):
        """
        Args:
            set_type: Register.ALL, Register.TRAIN, Register.TEST or list of them
            generator: would be returned that `True` for a generator or `False` for a list
        """
        if set_type == DataRegister.FULL:
            set_types = self.default_set_type
        elif isinstance(set_type, list):
            set_types = [_ for _ in set_type]
        elif isinstance(set_type, DataRegister):
            set_types = [set_type]
        else:
            raise ValueError(f'Unknown input {set_type = }')

        r = []
        for set_type in set_types:
            tmp = []
            if generator:
                r.append(self._call(set_type, **kwargs))

            else:
                for _ in tqdm(self._call(set_type, **kwargs)):
                    tmp.append(_)

                r.append(tmp)

        return r

    def _call(self, set_type, **kwargs):
        raise NotImplementedError


class DataSaver:
    pass
