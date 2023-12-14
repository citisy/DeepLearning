from .base import DataRegister, DataLoader


class Loader(DataLoader):
    """Recognizing Textual Entailment
    https://github.com/nyu-mll/GLUE-baselines

    Data structure:
        .
        ├── train.tsv   # 104743 items
        ├── dev.tsv     # 5463 items
        └── test.tsv    # 5461 items
    """

    default_set_type = [DataRegister.TRAIN, DataRegister.DEV, DataRegister.TEST]
    classes = ['not_entailment', 'entailment']

    def _call(self, set_type=DataRegister.TRAIN, **gen_kwargs):
        with open(f'{self.data_dir}/{set_type.value}.tsv', 'r', encoding='utf8') as f:
            gen_func = f.read().strip().split('\n')
        gen_func.pop(0)
        return self.gen_data(gen_func, set_type=set_type, **gen_kwargs)

    def get_ret(self, obj, set_type=DataRegister.TRAIN, **kwargs) -> dict:
        line = obj

        if set_type in {DataRegister.TRAIN, DataRegister.DEV}:
            index, sentence1, sentence2, label = line.split('\t')
            _class = self.classes.index(label)

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
