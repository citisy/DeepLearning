from .base import DataRegister, DataLoader


class Loader(DataLoader):
    """Quora Question Pairs
    https://github.com/nyu-mll/GLUE-baselines

    Data structure:
        .
        ├── original
        │   └── ...
        ├── train.tsv   # 363870 items
        ├── dev.tsv     # 40431 items
        └── test.tsv    # 390965 items
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
            _id, qid1, qid2, question1, question2, is_duplicate = line.split('\t')
            _class = int(is_duplicate)

            return dict(
                _id=(_id, qid1, qid2),
                _class=_class,
                texts=(question1, question2),
            )

        else:
            _id, question1, question2 = line.split('\t')

            return dict(
                _id=_id,
                texts=(question1, question2),
            )
