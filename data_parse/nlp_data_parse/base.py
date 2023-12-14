from tqdm import tqdm
from .. import DataRegister
from utils import os_lib, converter, visualize


class Token:
    """base google vocab token"""
    cls = '[CLS]'
    sep = '[SEP]'
    pad = '[PAD]'
    unk = '[UNK]'
    mask = '[MASK]'
    unused = '[unused%d]'


def fake_func(x):
    return x


class DataLoader:
    """
    for implementation, usually override the following methods:
        _call(): prepare the data, and return an iterable function warped by `gen_data()`
        get_ret(): logic of parsing the data, and return a dict of result
    """
    default_set_type = [DataRegister.TRAIN, DataRegister.TEST]
    default_data_type = DataRegister.FULL
    classes = []
    dataset_info: dict

    def __init__(self, data_dir, verbose=True, stdout_method=print, **kwargs):
        self.data_dir = data_dir
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else os_lib.FakeIo()
        self.__dict__.update(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.load(*args, **kwargs)

    def load(self, set_type=None, generator=True,
             use_multiprocess=False, n_process=5,
             **load_kwargs):
        """
        Args:
            set_type(list or DataRegister): a DataRegister type or a list of them
                FULL -> DataLoader.default_set_type
                other set_type -> [set_type]
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
            pbar = self._call(set_type=set_type, **load_kwargs)

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
        """prepare the data, and return an iterable function warped by `gen_data()`

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

    def gen_data(self, gen_func, max_size=None, **get_kwargs):
        """

        Args:
            gen_func:
            max_size: num of loaded data
            **get_kwargs:
                see also `get_ret` function to get more details of get_kwargs

        Yields
            a dict of result data

        """
        max_size = max_size or float('inf')
        i = 0
        for obj in gen_func:
            if i >= max_size:
                break

            obj = self.on_start_convert(obj)

            if not self.on_start_filter(obj):
                continue

            ret = self.get_ret(obj, **get_kwargs)
            if not ret:
                continue

            ret = self.on_end_convert(ret)

            if not self.on_end_filter(ret):
                continue

            i += 1
            yield ret

        if hasattr(gen_func, 'close'):
            gen_func.close()

    def get_ret(self, obj, **kwargs) -> dict:
        """logic of parsing the data, and return a dict of result"""
        raise NotImplementedError

    def on_start_convert(self, obj):
        return obj

    def on_end_convert(self, ret):
        return ret

    def on_start_filter(self, obj):
        return True

    def on_end_filter(self, ret):
        return True


class DataSaver:
    def __init__(self, data_dir, verbose=True, stdout_method=print):
        self.data_dir = data_dir
        self.default_set_type = [DataRegister.TRAIN, DataRegister.TEST]
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else os_lib.FakeIo()

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

        self.mkdirs(set_types=set_types, **kwargs)

        for iter_data, set_type in zip(data, set_types):
            if self.verbose:
                iter_data = tqdm(iter_data, desc=visualize.TextVisualize.highlight_str(f'Save {set_type.value} dataset'))

            self._call(iter_data, set_type=set_type, image_type=image_type, **kwargs)

    def mkdirs(self, **kwargs):
        os_lib.mk_dir(self.data_dir)

    def _call(self, iter_data, **gen_kwargs):
        raise NotImplementedError

    def gen_data(self, gen_func, max_size=float('inf'), **get_kwargs):
        """

        Args:
            gen_func:
            max_size: num of loaded data
            **get_kwargs:
                see also `parse_ret` function to get more details of get_kwargs

        Yields
            a dict of result data

        """
        i = 0
        for ret in gen_func:
            if i >= max_size:
                break

            ret = self.on_start_convert(ret)

            if not self.on_start_filter(ret):
                continue

            self.parse_ret(ret, **get_kwargs)
            i += 1

    def parse_ret(self, ret, image_type=DataRegister.PATH, **get_kwargs):
        raise NotImplementedError

    def on_start_convert(self, ret):
        return ret

    def on_start_filter(self, ret):
        return True

