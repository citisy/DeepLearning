from .base import DataRegister, DataLoader


class Loader(DataLoader):
    """Semantic Textual Similarity Benchmark
    https://github.com/nyu-mll/GLUE-baselines

    Data structure:
        .
        ├── original
        │   └── ...
        ├── train.tsv   # 5749 items
        ├── dev.tsv     # 1379 items
        └── test.tsv    # 1377 items
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
            index, genre, filename, year, old_index, source1, source2, sentence1, sentence2, score = line.split('\t')
            score = float(score)

            return dict(
                _id=index,
                genre=genre,
                score=score,
                texts=(sentence1, sentence2),
            )

        else:
            index, genre, filename, year, old_index, source1, source2, sentence1, sentence2 = line.split('\t')

            return dict(
                _id=index,
                genre=genre,
                texts=(sentence1, sentence2),
            )
