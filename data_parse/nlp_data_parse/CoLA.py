from .base import DataRegister, DataLoader


class Loader(DataLoader):
    """Corpus of Linguistic Acceptability
    https://gluebenchmark.com/tasks

    Data structure:
        .
        ├── original
        │   └── ...
        ├── train.tsv   # 8551 items
        ├── dev.tsv     # 1043 items
        └── test.tsv    # 1063 items
    """

    default_set_type = [DataRegister.TRAIN, DataRegister.DEV, DataRegister.TEST]

    def _call(self, set_type=DataRegister.TRAIN, **gen_kwargs):
        with open(f'{self.data_dir}/{set_type.value}.tsv', 'r', encoding='utf8') as f:
            gen_func = f.read().strip().split('\n')

        if set_type == DataRegister.TEST:
            gen_func.pop(0)

        gen_func = enumerate(gen_func)
        return self.gen_data(gen_func, set_type=set_type, **gen_kwargs)

    def get_ret(self, obj, set_type=DataRegister.TRAIN, **kwargs) -> dict:
        i, line = obj

        if set_type in {DataRegister.TRAIN, DataRegister.DEV}:
            _, _class, flag, text = line.split('\t', 3)
            _class = int(_class)

            return dict(
                _id=i,
                _class=_class,
                text=text,
                flag=flag
            )

        else:
            _, text = line.split('\t', 1)

            return dict(
                _id=i,
                text=text,
            )
