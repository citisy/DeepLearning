from .base import DataRegister, DataLoader, DataSaver
from utils import os_lib


class Loader(DataLoader):
    """

    Data structure:
        .
        └── corpus.txt
    """

    default_set_type = [DataRegister.MIX]

    def _call(self, **gen_kwargs):
        with open(f'{self.data_dir}/corpus', 'r', encoding='utf8') as f:
            gen_func = f.read().strip().split('\n')
        gen_func = enumerate(gen_func)
        return self.gen_data(gen_func, **gen_kwargs)

    def get_ret(self, obj, set_type=DataRegister.TRAIN, **kwargs) -> dict:
        i, line = obj
        context = line
        return dict(
            _id=i,
            context=context,
        )


class Saver(DataSaver):
    def mkdirs(self, **kwargs):
        os_lib.mk_dir(self.data_dir)

    def _call(self, iter_data, **gen_kwargs):
        f = open(f'{self.data_dir}/corpus.txt', 'w', encoding='utf8')
        return self.gen_data(iter_data, f=f, **gen_kwargs)

    def parse_ret(self, ret, f=None, **get_kwargs):
        context = ret['context']
        f.write(context + '\n')
