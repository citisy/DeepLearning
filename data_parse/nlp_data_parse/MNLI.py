from .base import DataRegister, DataLoader


class Loader(DataLoader):
    """Multi-Genre Natural Language Inference Corpus
    https://github.com/nyu-mll/GLUE-baselines

    Data structure:
        .
        ├── original
        │   └── ...
        ├── train.tsv               # 392702 items
        ├── dev_matched.tsv         # same source, 9815 items
        ├── dev_mismatched.tsv      # different source, 9832 items
        ├── test_matched.tsv        # same source, 9796 items
        └── test_mismatched.tsv     # different source, 9847 items
    """

    default_set_type = [DataRegister.TRAIN, DataRegister.DEV, DataRegister.TEST]
    classes = ['entailment', 'contradiction', 'neutral']

    def _call(self, set_type=DataRegister.TRAIN, task='full', **gen_kwargs):
        """

        Args:
            set_type:
            task: 'matched', 'mismatched' or 'full', do not affect in train mode

        """
        if set_type == DataRegister.TRAIN:
            with open(f'{self.data_dir}/{set_type.value}.tsv', 'r', encoding='utf8') as f:
                gen_func = f.read().strip().split('\n')
            gen_func.pop(0)

        else:
            if task == 'full':
                tasks = ['matched', 'mismatched']
            else:
                tasks = [task]

            gen_func = []
            for t in tasks:
                with open(f'{self.data_dir}/{set_type.value}_{t}.tsv', 'r', encoding='utf8') as f:
                    tmp = f.read().strip().split('\n')
                tmp.pop(0)
                gen_func += tmp

        return self.gen_data(gen_func, set_type=set_type, **gen_kwargs)

    def get_ret(self, obj, set_type=DataRegister.TRAIN, **kwargs) -> dict:
        line = obj

        if set_type in {DataRegister.TRAIN, DataRegister.DEV}:
            if set_type == DataRegister.TRAIN:
                index, promptID, pairID, genre, sentence1_binary_parse, sentence2_binary_parse, sentence1_parse, sentence2_parse, sentence1, sentence2, label1, gold_label = line.split('\t')
                aux_class = (label1,)
            else:
                index, promptID, pairID, genre, sentence1_binary_parse, sentence2_binary_parse, sentence1_parse, sentence2_parse, sentence1, sentence2, label1, label2, label3, label4, label5, gold_label = line.split('\t')
                aux_class = (label1, label2, label3, label4, label5)

            aux_class = [self.classes.index(i) for i in aux_class]
            gold_label = self.classes.index(gold_label)
            return dict(
                _id=(index, promptID, pairID),
                _class=gold_label,
                aux_class=aux_class,
                genre=genre,
                texts=(sentence1, sentence2),
                texts_parse=(sentence1_parse, sentence2_parse),
                texts_binary_parse=(sentence1_binary_parse, sentence2_binary_parse)
            )

        else:
            index, promptID, pairID, genre, sentence1_binary_parse, sentence2_binary_parse, sentence1_parse, sentence2_parse, sentence1, sentence2 = line.split('\t')

            return dict(
                _id=(index, promptID, pairID),
                genre=genre,
                texts=(sentence1, sentence2),
                texts_parse=(sentence1_parse, sentence2_parse),
                texts_binary_parse=(sentence1_binary_parse, sentence2_binary_parse)
            )
