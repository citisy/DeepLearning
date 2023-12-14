from .base import DataRegister, DataLoader


class Loader(DataLoader):
    """Winograd NLI, reading comprehension
    https://github.com/nyu-mll/GLUE-baselines

    Data structure:
        .
        ├── train.tsv   # 635 items
        ├── dev.tsv     # 71 items
        └── test.tsv    # 146 items
    """

    default_set_type = [DataRegister.TRAIN, DataRegister.DEV, DataRegister.TEST]

    def _call(self, set_type=DataRegister.TRAIN, **gen_kwargs):
        with open(f'{self.data_dir}/{set_type.value}.tsv', 'r', encoding='utf8') as f:
            gen_func = f.read().strip().split('\n')
        gen_func.pop(0)
        return self.gen_data(gen_func, set_type=set_type, **gen_kwargs)

    def get_ret(self, obj, set_type=DataRegister.TRAIN, **kwargs) -> dict:
        line = obj

        if set_type in {DataRegister.TRAIN, DataRegister.DEV}:
            index, sentence1, sentence2, label = line.split('\t')
            _class = int(label)

            return dict(
                _id=index,
                _class=_class,
                texts=(sentence1, sentence2),
            )

        else:
            index, sentence1, sentence2 = line.split('\t')

            return dict(
                _id=index,
                texts=(sentence1, sentence2),
            )
