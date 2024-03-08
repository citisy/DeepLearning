import torch
import numpy as np
from .text_pretrain import TextProcess, Bert as BertFull, FromHFPretrain


class CoLA(TextProcess):
    dataset_version = 'CoLA'
    data_dir = 'data/CoLA'

    # mean seq len is 9.327213191439597, max seq len is 45, min seq len is 2
    max_seq_len = 64
    n_classes = 2

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nlp_data_parse.CoLA import Loader, DataRegister
        loader = Loader(self.data_dir)
        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]
        else:
            return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class SST2(TextProcess):
    dataset_version = 'SST2'
    data_dir = 'data/SST2'

    # mean seq len is 11.319262349849293, max seq len is 64, min seq len is 1
    max_seq_len = 64
    n_classes = 2

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nlp_data_parse.SST2 import Loader, DataRegister
        loader = Loader(self.data_dir)
        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]
        else:
            return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class Bert(BertFull):
    is_mlm = False  # only nsp strategy

    def set_model(self):
        from models.text_classification.bert import Model
        self.get_vocab()
        self.model = Model(self.vocab_op.vocab_size, sp_tag_dict=self.vocab_op.sp_tag_dict, out_features=self.n_classes)

    def set_optimizer(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)


class McMetric:
    def metric(self, *args, **kwargs) -> dict:
        """use Matthew's Corr"""
        from metrics import classification
        container = self.predict(**kwargs)

        metric_results = {}
        for name, results in container['model_results'].items():
            next_trues = np.array(results['next_trues'])
            next_preds = np.array(results['next_preds'])
            result = classification.pr.mcc(next_trues, next_preds)

            result.update(
                score=result['mcc']
            )

            metric_results[name] = result

        return metric_results


class Bert_CoLA(McMetric, Bert, CoLA):
    """
    Usage:
        .. code-block:: python

            from examples.text_classification import Bert_CoLA as Process

            # about 200M data pretrain
            # it seems that the pretraining model has significantly influenced the score
            Process(pretrain_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.10233}  # Matthew's Corr
    """


class BertHF_CoLA(McMetric, Bert, FromHFPretrain, CoLA):
    """
    Usage:
        .. code-block:: python

            from examples.text_classification import BertHF_CoLA as Process

            # if using `bert-base-uncased` pretrain model
            Process(pretrain_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.5481}  # Matthew's Corr
            # benchmark: 0.5653
    """


class Bert_SST2(Bert, SST2):
    """
    Usage:
        .. code-block:: python

            from examples.text_classification import Bert_SST2 as Process

            # no pretrain data, use SST2 data to train directly
            Process(vocab_fn='...').run(max_epoch=100, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.78899}     # acc

            # about 200M data pretrain
            Process(pretrain_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.83142}     # acc
    """


class BertFull_SST2(BertFull, SST2):
    """
    Usage:
        .. code-block:: python

            from examples.text_classification import BertFull_SST2 as Process

            # no pretrain data, use SST2 data to train with nsp and mlm directly
            Process(vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.78096}     # acc
    """


class BertHF_SST2(Bert, FromHFPretrain, SST2):
    """
    Usage:
        .. code-block:: python

            from examples.text_classification import BertHF_SST2 as Process

            # if using `bert-base-uncased` pretrain model
            Process(pretrain_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.92316}   # acc
            # benchmark: 0.9232
    """
