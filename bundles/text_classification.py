from typing import List

import torch
import numpy as np

from data_parse.nl_data_parse.pre_process import spliter
from data_parse.nl_data_parse.pre_process.bundled import SimpleTokenizer
from utils import torch_utils
from .text_pretrain import TextProcessForBert, BaseBert as BertFull, FromBertHFPretrained
from processor import Process


class CoLA(TextProcessForBert):
    dataset_version = 'CoLA'
    data_dir = 'data/CoLA'

    # mean seq len is 9.327213191439597, max seq len is 45, min seq len is 2
    max_seq_len = 64
    n_classes = 2

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nl_data_parse.datasets.CoLA import Loader, DataRegister
        loader = Loader(self.data_dir)
        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]
        else:
            return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class SST2(TextProcessForBert):
    dataset_version = 'SST2'
    data_dir = 'data/SST2'

    # mean seq len is 11.319262349849293, max seq len is 64, min seq len is 1
    max_seq_len = 64
    n_classes = 2

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nl_data_parse.datasets.SST2 import Loader, DataRegister
        loader = Loader(self.data_dir)
        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]
        else:
            return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]


class Bert(BertFull):
    is_token_cls = False  # only nsp strategy

    def set_model(self):
        from models.text_classification.bert import Model
        self.model = Model(self.tokenizer.vocab_size, pad_id=self.tokenizer.pad_id, out_features=self.n_classes)

    def set_optimizer(self, lr=5e-5, **kwargs):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)


class McMetric:
    def metric(self, *args, **kwargs) -> dict:
        """use Matthew's Corr"""
        from metrics import classification
        container = self.predict(**kwargs)

        metric_results = {}
        for name, results in container['model_results'].items():
            seq_cls_trues = np.array(results['seq_cls_trues'])
            seq_cls_preds = np.array(results['seq_cls_preds'])
            result = classification.pr.mcc(seq_cls_trues, seq_cls_preds)

            result.update(
                score=result['mcc']
            )

            metric_results[name] = result

        return metric_results


class Bert_CoLA(McMetric, Bert, CoLA):
    """
    Usage:
        .. code-block:: python

            from bundles.text_classification import Bert_CoLA as Process

            # about 200M data pretrain
            # it seems that the pretraining model has significantly influenced the score
            Process(pretrained_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.10233}  # Matthew's Corr
    """


class BertHF_CoLA(McMetric, Bert, FromBertHFPretrained, CoLA):
    """
    Usage:
        .. code-block:: python

            from bundles.text_classification import BertHF_CoLA as Process

            # if using `bert-base-uncased` pretrain model
            Process(pretrained_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.5481}  # Matthew's Corr
            # benchmark: 0.5653
    """


class Bert_SST2(Bert, SST2):
    """
    Usage:
        .. code-block:: python

            from bundles.text_classification import Bert_SST2 as Process

            # no pretrain data, use SST2 data to train directly
            Process(vocab_fn='...').run(max_epoch=100, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.78899}     # acc

            # about 200M data pretrain
            Process(pretrained_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.83142}     # acc
    """


class BertFull_SST2(BertFull, SST2):
    """
    Usage:
        .. code-block:: python

            from bundles.text_classification import BertFull_SST2 as Process

            # no pretrain data, use SST2 data to train with nsp and mlm directly
            Process(vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.78096}     # acc
    """


class BertHF_SST2(Bert, FromBertHFPretrained, SST2):
    """
    Usage:
        .. code-block:: python

            from bundles.text_classification import BertHF_SST2 as Process

            # if using `bert-base-uncased` pretrain model
            Process(pretrained_model='...', vocab_fn='...').run(max_epoch=5, train_batch_size=128, fit_kwargs=dict(check_period=1))
            {'score': 0.92316}   # acc
            # benchmark: 0.9232
    """


class CTTransformer(Process):
    model_version = 'CTTransformer'

    def set_model(self):
        from models.text_classification.CTTransformer import Model
        self.model = Model()

    def set_tokenizer(self):
        self.tokenizer = SimpleTokenizer.from_pretrained(
            self.vocab_fn,
            sp_token_dict=dict(
                # blank='<blank>',
                # sos='<s>',
                # eos='</s>',
                unk='<unk>',
                ignore='<unk>'
            )
        )

    def load_pretrained(self):
        from models.text_classification.CTTransformer import WeightConverter
        tensor = torch.load(self.pretrained_model, map_location=torch.device('cpu'))
        tensor = WeightConverter.from_official(tensor)

        self.model.load_state_dict(tensor, strict=True)

    def get_model_inputs(self, loop_inputs, train=True):
        segments = [ret['segment'] for ret in loop_inputs]
        segments_ids = self.tokenizer.encode_segments(segments)
        segments_ids = torch_utils.Converter.force_to_tensors(segments_ids, self.device)
        inputs = dict(
            text_ids=segments_ids,
            segments=segments,
        )
        return inputs

    def on_val_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)
        model_inputs.update(model_kwargs)

        model_results = {}
        for name, model in self.models.items():
            model_output = model(**model_inputs)
            model_results[name] = model_output
            model_results[name]['segments'] = model_inputs['segments']

        return model_results

    punc_list = ("，", "。", "？", "、")

    def decode_segment(self, pred, segment):
        new_segment = []
        for punc_id, word in zip(pred, segment):
            new_segment.append(word)
            if punc_id > 1:
                new_segment.append(self.punc_list[punc_id - 2])

        return new_segment

    def on_val_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        preds = model_results[self.model_name]['preds']
        segments = model_results[self.model_name]['segments']

        results = []
        for punc_ids, segment in zip(preds, segments):
            segment = self.decode_segment(punc_ids, segment)
            results.append(dict(
                punc_ids=punc_ids,
                segment=segment,
            ))
        process_results.setdefault(self.model_name, []).extend(results)

    def gen_predict_inputs(
            self, *objs, start_idx=None, end_idx=None,
            segments=None,
            **kwargs
    ) -> List[dict]:
        if isinstance(segments, List) and isinstance(segments[0], str):
            segments = [None] * start_idx + [segments] * (end_idx - start_idx)

        inputs = [
            dict(
                segment=segments[i],
            )
            for i in range(start_idx, end_idx)
        ]

        return inputs

    def on_predict_reprocess(self, loop_objs, **kwargs):
        self.on_val_reprocess(loop_objs, **kwargs)
