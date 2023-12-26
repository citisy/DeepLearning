import torch
import numpy as np
from processor import BaseDataset
from utils import os_lib
from data_parse.nlp_data_parse.pre_process import spliter, encoder, bundled, sp_token_dict, dict_maker
import pandas as pd
import re
from .text_pretrain import DataProcess as DataHooks, Bert as Process


class DataProcess(DataHooks):
    def make_vocab(self):
        # todo: make word piece
        def filter_func(x):
            if re.search('[0-9]', x):
                return False

            if re.search('[^a-z]', x):
                return False

            return True

        iter_data = self.get_train_data()
        paragraphs = [ret['text'] for ret in iter_data]
        paragraphs = bundled.lower(paragraphs)
        segments = spliter.segments_from_paragraphs(paragraphs)
        word_dict = dict_maker.word_id_dict(segments, start_id=len(sp_token_dict), filter_func=filter_func)
        vocab = list(sp_token_dict.values()) + list(word_dict.keys())
        self.save_vocab(vocab)
        return vocab

    vocab: set

    def train_data_preprocess(self, iter_data):
        paragraphs = [ret['text'] for ret in iter_data]
        paragraphs = bundled.lower(paragraphs)
        segments = spliter.segments_from_paragraphs(paragraphs, is_word_piece=True, vocab=self.vocab)
        for ret, segment in zip(iter_data, segments):
            ret.update(
                segment=segment,
                segment_tag=[1] * len(segment)
            )
        return iter_data

    def count_seq_len(self):
        iter_data = self.get_train_data()
        iter_data = self.train_data_preprocess(iter_data)
        s = [len(ret['segment']) for ret in iter_data]
        self.log(f'seq_len = {np.mean(s)}')


class SST2(DataProcess):
    dataset_version = 'SST2'

    val_dataset_ins = BaseDataset
    train_dataset_ins = BaseDataset

    train_data_num = None
    val_data_num = None

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

    def train_data_augment(self, ret) -> dict:
        segments = [ret['segment']]
        segment_tags = [ret['segment_tag']]
        segments, mask_tags = bundled.random_mask(segments, self.word_dict, mask_token=sp_token_dict['mask'], unk_tag=self.sp_tag_dict['unk'], non_mask_tag=self.sp_tag_dict['non_mask'])

        segments = bundled.align(segments, seq_len=self.seq_len, start_token=sp_token_dict['cls'], end_token=sp_token_dict['sep'], pad_token=sp_token_dict['pad'])
        segment_tags = bundled.align(segment_tags, seq_len=self.seq_len, start_token=1, end_token=1, pad_token=self.sp_tag_dict['seg_pad'])
        mask_tags = bundled.align(mask_tags, seq_len=self.seq_len, start_token=self.sp_tag_dict['non_mask'], end_token=self.sp_tag_dict['non_mask'], pad_token=self.sp_tag_dict['non_mask'])
        text_tags = encoder.simple(segments, self.word_dict, unk_tag=self.sp_tag_dict['unk'])

        return dict(
            segment=segments[0],
            mask_tag=mask_tags[0],
            text_tag=text_tags[0],
            segment_tag=segment_tags[0],
            _class=ret['_class'],
            ori_text=ret['text']
        )

    def val_data_augment(self, ret) -> dict:
        segments = [ret['segment']]
        segment_tags = [ret['segment_tag']]
        segments = bundled.align(segments, seq_len=self.seq_len, start_token=sp_token_dict['cls'], end_token=sp_token_dict['sep'], pad_token=sp_token_dict['pad'])
        segment_tags = bundled.align(segment_tags, seq_len=self.seq_len, start_token=1, end_token=1, pad_token=self.sp_tag_dict['seg_pad'])
        text_tags = encoder.simple(segments, self.word_dict, unk_tag=self.sp_tag_dict['unk'])

        return dict(
            segment=segments[0],
            text_tag=text_tags[0],
            segment_tag=segment_tags[0],
            _class=ret['_class'],
            ori_text=ret['text']
        )


class Bert(Process):
    train_pretrain = False
    pretrain_model: str

    def set_model(self):
        self.get_vocab()

        if self.train_pretrain:
            from models.text_pretrain.bert import Model

            self.model = Model(self.vocab_size, seq_len=self.seq_len, sp_tag_dict=self.sp_tag_dict, n_segment=1)

        else:
            from models.text_classification.bert import Model

            self.model = Model(self.vocab_size, seq_len=self.seq_len, sp_tag_dict=self.sp_tag_dict)
            ckpt = torch.load(self.pretrain_model, map_location=self.device)
            self.model.load_state_dict(ckpt['model'], strict=False)

    def on_val_reprocess(self, rets, model_results, container, **kwargs):
        for name, results in model_results.items():
            r = container['model_results'].setdefault(name, dict())
            r.setdefault('trues', []).extend([ret['_class'] for ret in rets])
            r.setdefault('preds', []).extend(results['preds'])
            r.setdefault('text', []).extend([ret['ori_text'] for ret in rets])

    def on_val_end(self, container, is_visualize=False, **kwargs):
        if is_visualize:
            for name, results in container['model_results'].items():
                data = [dict(
                    true=true,
                    pred=pred,
                    text=text,
                ) for text, true, pred in zip(results['text'], results['trues'], results['preds'])]
                df = pd.DataFrame(data)
                os_lib.Saver(stdout_method=self.log).auto_save(df, f'{self.cache_dir}/{self.counters["epoch"]}/{name}.csv', index=False)


class Bert_SST2(Bert, SST2):
    """
    Usage:
        .. code-block:: python

            from examples.text_pair_classification import Bert_SST2 as Process

            Process(train_pretrain=True).run(max_epoch=100, train_batch_size=128, predict_batch_size=128, check_period=3)
            {'score': 0.78096}  # no pretrain data, use SST2 data to train directly
    """
