from .base import DataRegister, DataLoader


class Loader(DataLoader):
    """The Stanford Sentiment Treebank
    https://github.com/nyu-mll/GLUE-baselines

    Data structure:
        .
        ├── original
        │   └── ...
        ├── train.tsv   # 67350 items
        ├── dev.tsv     # 873 items
        └── test.tsv    # 1821 items
    """

    default_set_type = [DataRegister.TRAIN, DataRegister.DEV, DataRegister.TEST]
    classes = ['neg', 'pos']

    def _call(self, set_type=DataRegister.TRAIN, **gen_kwargs):
        with open(f'{self.data_dir}/{set_type.value}.tsv', 'r', encoding='utf8') as f:
            gen_func = f.read().strip().split('\n')
        gen_func.pop(0)
        gen_func = enumerate(gen_func)
        return self.gen_data(gen_func, set_type=set_type, **gen_kwargs)

    def get_ret(self, obj, set_type=DataRegister.TRAIN, **kwargs) -> dict:
        i, line = obj

        if set_type in {DataRegister.TRAIN, DataRegister.DEV}:
            context, _class = line.split('\t', 1)
            _class = int(_class)
            context = context.strip()

            return dict(
                _id=i,
                _class=_class,
                context=context,
            )

        else:
            _, context = line.split('\t', 1)

            return dict(
                _id=i,
                context=context,
            )
