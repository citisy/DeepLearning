from .base import DataRegister, DataLoader, DataSaver
from utils import os_lib


class Loader(DataLoader):
    """

    Data structure:
        .
        └── corpus.txt
    """

    default_set_type = [DataRegister.MIX]

    def _call(self, set_type=DataRegister.TRAIN, **gen_kwargs):
        with open(f'{self.data_dir}/{set_type.value}.txt', 'r', encoding='utf8') as f:
            gen_func = f.read().strip().split('\n')
        gen_func = enumerate(gen_func)
        return self.gen_data(gen_func, **gen_kwargs)

    def get_ret(self, obj, set_type=DataRegister.TRAIN, return_label=True, **kwargs) -> dict:
        i, line = obj
        ret = dict(_id=i)

        if return_label:
            _class, text = line
            ret.update(_class=_class, text=text)
        else:
            text = line
            ret.update(text=text)

        return ret


class Saver(DataSaver):
    def mkdirs(self, **kwargs):
        os_lib.mk_dir(self.data_dir)

    def _call(self, iter_data, set_type=DataRegister.TRAIN, **gen_kwargs):
        f = open(f'{self.data_dir}/{set_type.value}.txt', 'w', encoding='utf8')
        return self.gen_data(iter_data, f=f, **gen_kwargs)

    def parse_ret(self, ret, f=None, return_label=True, **get_kwargs):
        text = ret['text']
        if return_label:
            _class = ret['_class']
            f.write(f'{_class}\t{text}\n')
        else:
            f.write(text + '\n')
