import torch
from .text_pretrain import TextPairProcessForBert, BaseBert as BertFull, FromBertHFPretrained


class MNLI(TextPairProcessForBert):
    dataset_version = 'MNLI'
    data_dir = 'data/MNLI'

    # mean seq len is 36.91200452251326, max seq len is 441, min seq len is 2
    max_seq_len = 64
    n_classes = 3

    def get_data(self, *args, train=True, task='matched', **kwargs):
        from data_parse.nl_data_parse.datasets.MNLI import Loader, DataRegister
        loader = Loader(self.data_dir)

        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]
        else:
            return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False, task=task)[0]


class QQP(TextPairProcessForBert):
    dataset_version = 'QQP'
    data_dir = 'data/QQP'

    # mean seq len is 27.579311027192823, max seq len is 327, min seq len is 3
    max_seq_len = 64
    n_classes = 2

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nl_data_parse.datasets.QQP import Loader, DataRegister
        loader = Loader(self.data_dir)

        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]
        else:
            return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class QNLI(TextPairProcessForBert):
    dataset_version = 'QNLI'
    data_dir = 'data/QNLI'

    # mean seq len is 46.41800406709756, max seq len is 545, min seq len is 9
    max_seq_len = 64
    n_classes = 2

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nl_data_parse.datasets.QNLI import Loader, DataRegister
        loader = Loader(self.data_dir)
        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]
        else:
            return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class MRPC(TextPairProcessForBert):
    dataset_version = 'MRPC'
    data_dir = 'data/MRPC'

    # mean seq len is 50.24165848871443, max seq len is 100, min seq len is 16
    max_seq_len = 128
    n_classes = 2

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nl_data_parse.datasets.MRPC import Loader, DataRegister
        loader = Loader(self.data_dir)
        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]
        else:
            return loader.load(set_type=DataRegister.TEST, max_size=self.val_data_num, generator=False)[0]


class RTE(TextPairProcessForBert):
    dataset_version = 'RTE'
    data_dir = 'data/RTE'

    # mean seq len is 67.19799196787149, max seq len is 286, min seq len is 10
    max_seq_len = 128
    n_classes = 2

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nl_data_parse.datasets.RTE import Loader, DataRegister
        loader = Loader(self.data_dir)
        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]
        else:
            return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class Bert(BertFull):
    is_token_cls = False  # only nsp strategy

    def set_model(self):
        from models.text_pair_classification.bert import Model

        self.model = Model(self.tokenizer.vocab_size, pad_id=self.tokenizer.pad_id, out_features=self.n_classes)

    def set_optimizer(self, lr=5e-5, **kwargs):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)


class Bert_MNLI(Bert, MNLI):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pair_classification import Bert_MNLI as Process

            # about 200M data pretrain
            process = Process(pretrained_model='...', vocab_fn='...')
            process.fit(max_epoch=5, batch_size=128, dataloader_kwargs=dict(num_workers=8))

            process.metric(batch_size=128, data_get_kwargs=dict(task='matched'))
            {'score': 0.6782}  # match acc

            process.metric(batch_size=128, data_get_kwargs=dict(task='mismatched'))
            {'score': 0.6891}  # mismatch acc
    """


class BertHF_MNLI(Bert, FromBertHFPretrained, MNLI):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pair_classification import BertHF_MNLI as Process

            # if using `bert-base-uncased` pretrain model
            process = Process(pretrained_model='...', vocab_fn='...')
            process.init()
            process.fit(max_epoch=5, batch_size=128, dataloader_kwargs=dict(num_workers=8))

            process.metric(batch_size=128, data_get_kwargs=dict(task='matched'))
            {'score': 0.8202}  # match acc
            # benchmark: 0.8391

            process.metric(batch_size=128, data_get_kwargs=dict(task='mismatched'))
            {'score': 0.8210}  # mismatch acc
            # benchmark: 0.8410
    """


class Bert_QQP(Bert, QQP):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pair_classification import Bert_QQP as Process

            # about 200M data pretrain
            Process(pretrained_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, predict_batch_size=128, check_period=1)
            {'score': 0.86450/0.82584}    # acc/f1
    """


class BertHF_QQP(Bert, FromBertHFPretrained, QQP):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pair_classification import BertHF_QQP as Process

            # if using `bert-base-uncased` pretrain model
            Process(pretrained_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.9117/0.8803}    # acc/f1
            # benchmark: 0.9071/0.8749
    """


class Bert_QNLI(Bert, QNLI):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pair_classification import Bert_QNLI as Process

            # no pretrain data, use QNLI data to train directly
            Process(vocab_fn='...').run(max_epoch=50, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.5712}   # acc

            # about 200M data pretrain
            Process(pretrained_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.8002}   # acc

    """


class BertFull_QNLI(BertFull, QNLI):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pair_classification import BertFull_QNLI as Process

            # no pretrain data, use QNLI data to train with nsp and mlm directly
            Process(vocab_fn='...').run(max_epoch=50, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.80395}   # acc
    """


class BertHF_QNLI(Bert, FromBertHFPretrained, QNLI):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pair_classification import BertHF_QNLI as Process

            # if using `bert-base-uncased` pretrain model
            Process(pretrained_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.89254}   # acc
            # benchmark: 0.9066
    """


class Bert_MRPC(Bert, MRPC):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pair_classification import Bert_QNLI as Process

            # about 200M data pretrain
            Process(pretrained_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.71420/0.80032}   # acc/f1
    """


class BertHF_MRPC(Bert, FromBertHFPretrained, MRPC):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pair_classification import BertHF_MRPC as Process

            # if using `bert-base-uncased` pretrain model
            Process(pretrained_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.8521/0.8913}   # acc/f1
            # benchmark: 0.8407/0.8885
    """


class Bert_RTE(Bert, RTE):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pair_classification import Bert_RTE as Process

            # about 200M data pretrain
            Process(pretrained_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.57761}   # acc
    """


class BertHF_RTE(Bert, FromBertHFPretrained, RTE):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pair_classification import BertHF_RTE as Process

            # if using `bert-base-uncased` pretrain model
            Process(pretrained_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.6859}   # acc
            # benchmark: 0.6570
    """
