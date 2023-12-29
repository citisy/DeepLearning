import torch
import numpy as np
from .text_pretrain import TextProcess, BertNSP


class CoLA(TextProcess):
    dataset_version = 'CoLA'
    data_dir = 'data/CoLA'
    seq_len = 16  # mean seq_len = 11.95

    def get_train_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.CoLA import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]

    def get_val_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.CoLA import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class SST2(TextProcess):
    dataset_version = 'SST2'
    data_dir = 'data/SST2'
    seq_len = 16  # mean seq_len = 11.95

    def get_train_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.SST2 import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]

    def get_val_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.SST2 import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class Bert(BertNSP):
    pretrain_model: str

    def set_model(self):
        from models.text_classification.bert import Model
        self.get_vocab()
        self.model = Model(self.vocab_size, seq_len=self.seq_len, sp_tag_dict=self.sp_tag_dict)

        if hasattr(self, 'pretrain_model'):
            ckpt = torch.load(self.pretrain_model, map_location=self.device)
            self.model.load_state_dict(ckpt['model'], strict=False)

    def set_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5, betas=(0.5, 0.999))


class Bert_CoLA(Bert, CoLA):
    """
    Usage:
        .. code-block:: python

            from examples.text_pair_classification import Bert_SST2 as Process

            Process(train_pretrain=True).run(max_epoch=100, train_batch_size=128, predict_batch_size=128, check_period=3)
            {'score': 0.78096}  # no pretrain data, use SST2 data to train directly
    """

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


class Bert_SST2(Bert, SST2):
    """
    Usage:
        .. code-block:: python

            from examples.text_pair_classification import Bert_SST2 as Process

            Process(train_pretrain=True).run(max_epoch=100, train_batch_size=128, predict_batch_size=128, check_period=3)
            {'score': 0.78096}  # no pretrain data, use SST2 data to train directly
    """
