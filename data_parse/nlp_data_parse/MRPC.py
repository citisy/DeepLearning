from .base import DataRegister, DataLoader


class Loader(DataLoader):
    """Microsoft Research Paraphrase Corpus
    https://github.com/nyu-mll/GLUE-baselines

    Data structure:
        .
        ├── msr_paraphrase_data.txt     # single text, 10948 items
        ├── msr_paraphrase_test.txt     # similar text pair, 1725 items
        └── msr_paraphrase_train.txt    # similar text pair, 4076 items
    """
    default_set_type = [DataRegister.TRAIN, DataRegister.TEST]

    def _call(self, set_type=DataRegister.TRAIN, **gen_kwargs):
        with open(f'{self.data_dir}/msr_paraphrase_{set_type.value}.txt', 'r', encoding='utf8') as f:
            gen_func = f.read().strip().split('\n')

        gen_func.pop(0)
        return self.gen_data(gen_func, **gen_kwargs)

    def get_ret(self, obj, **kwargs) -> dict:
        line = obj
        _class, id1, id2, text1, text2 = line.split('\t')
        _class = int(_class)

        return dict(
            _id=(id1, id2),
            _class=_class,
            texts=(text1, text2),
        )
