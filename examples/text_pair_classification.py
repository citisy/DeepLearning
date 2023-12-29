import torch
from .text_pretrain import TextPairProcess, BertNSP


class MNLI(TextPairProcess):
    dataset_version = 'MNLI'
    data_dir = 'data/MNLI'
    seq_len = 64  # mean seq_len = 46.418

    def get_train_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.MNLI import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]

    def get_val_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.MNLI import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class QQP(TextPairProcess):
    dataset_version = 'QQP'
    data_dir = 'data/QQP'
    seq_len = 64  # mean seq_len = 46.418

    def get_train_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.QQP import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]

    def get_val_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.QQP import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class QNLI(TextPairProcess):
    dataset_version = 'QNLI'
    data_dir = 'data/QNLI'
    seq_len = 64  # mean seq_len = 46.418

    def get_train_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.QNLI import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]

    def get_val_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.QNLI import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class STSB(TextPairProcess):
    dataset_version = 'STSB'
    data_dir = 'data/STSB'
    seq_len = 64  # mean seq_len = 46.418

    def get_train_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.STSB import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]

    def get_val_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.STSB import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class MRPC(TextPairProcess):
    dataset_version = 'MRPC'
    data_dir = 'data/MRPC'
    seq_len = 64  # mean seq_len = 46.418

    def get_train_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.MRPC import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]

    def get_val_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.MRPC import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class RTE(TextPairProcess):
    dataset_version = 'RTE'
    data_dir = 'data/RTE'
    seq_len = 64  # mean seq_len = 46.418

    def get_train_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.RTE import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]

    def get_val_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.RTE import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class Bert(BertNSP):
    pretrain_model: str

    def set_model(self):
        from models.text_pair_classification.bert import Model

        self.get_vocab()
        self.model = Model(self.vocab_size, seq_len=self.seq_len, sp_tag_dict=self.sp_tag_dict)
        if hasattr(self, 'pretrain_model'):
            ckpt = torch.load(self.pretrain_model, map_location=self.device)
            self.model.load_state_dict(ckpt['model'], strict=False)


class Bert_MNLI(Bert, MNLI):
    """
    Usage:
        .. code-block:: python

            from examples.text_pair_classification import Bert_QNLI as Process

            Process(train_pretrain=True).run(max_epoch=100, train_batch_size=128, predict_batch_size=128, check_period=3)
            {'score': 0.80395}  # no pretrain data, use QNLI data to train directly
    """


class Bert_QQP(Bert, QQP):
    """
    Usage:
        .. code-block:: python

            from examples.text_pair_classification import Bert_QNLI as Process

            Process(train_pretrain=True).run(max_epoch=100, train_batch_size=128, predict_batch_size=128, check_period=3)
            {'score': 0.80395}  # no pretrain data, use QNLI data to train directly
    """


class Bert_QNLI(Bert, QNLI):
    """
    Usage:
        .. code-block:: python

            from examples.text_pair_classification import Bert_QNLI as Process

            Process(train_pretrain=True).run(max_epoch=100, train_batch_size=128, predict_batch_size=128, check_period=3)
            {'score': 0.80395}  # no pretrain data, use QNLI data to train directly
    """


class Bert_STSB(Bert, STSB):
    """
    Usage:
        .. code-block:: python

            from examples.text_pair_classification import Bert_QNLI as Process

            Process(train_pretrain=True).run(max_epoch=100, train_batch_size=128, predict_batch_size=128, check_period=3)
            {'score': 0.80395}  # no pretrain data, use QNLI data to train directly
    """


class Bert_MRPC(Bert, MRPC):
    """
    Usage:
        .. code-block:: python

            from examples.text_pair_classification import Bert_QNLI as Process

            Process(train_pretrain=True).run(max_epoch=100, train_batch_size=128, predict_batch_size=128, check_period=3)
            {'score': 0.80395}  # no pretrain data, use QNLI data to train directly
    """


class Bert_RTE(Bert, RTE):
    """
    Usage:
        .. code-block:: python

            from examples.text_pair_classification import Bert_QNLI as Process

            Process(train_pretrain=True).run(max_epoch=100, train_batch_size=128, predict_batch_size=128, check_period=3)
            {'score': 0.80395}  # no pretrain data, use QNLI data to train directly
    """
