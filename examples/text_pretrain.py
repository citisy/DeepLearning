import torch
import numpy as np
from processor import Process, DataHooks, BaseDataset
from utils import math_utils, os_lib
from data_parse.nlp_data_parse.pre_process import spliter, encoder, bundled, sp_token_dict, dict_maker
import pandas as pd
import re


class RandomChoiceTextPairsDataset(BaseDataset):
    """for Next Sentence Prediction"""

    def __getitem__(self, idx):
        """all text pair in iter_data is the true text pair"""
        ret = self.iter_data[idx]
        texts = ret['texts']
        segments = ret['segments']
        segment_tags = ret['segment_tags']

        # 50% to select another text as the false sample
        if np.random.random() < 0.5:
            next_ret = np.random.choice(self.iter_data)
            next_text = next_ret['texts'][1]
            next_segment = next_ret['segments'][1]
            next_segment_tag = next_ret['segment_tags'][1]

            texts = (texts[0], next_text)
            segments = (segments[0], next_segment)
            segment_tags = (segment_tags[0], next_segment_tag)
            _class = 0

        else:
            _class = 1

        ret = dict(
            texts=texts,
            segments=segments,
            segment_tags=segment_tags,
            _class=_class
        )

        return self.augment_func(ret)


class RandomReverseTextPairsDataset(BaseDataset):
    """for Sentence Order Prediction"""

    def __getitem__(self, idx):
        ret = self.iter_data[idx]
        text = ret['text']
        segment = ret['segment']
        segment_tag = ret['segment_tag']

        # 50% to reverse the text
        if np.random.random() < 0.5:
            next_segment = segment[::-1]
            _class = 0
        else:
            next_segment = segment
            _class = 1

        ret = dict(
            texts=(text, text),
            segment_pair=(segment, next_segment),
            segment_tag_pairs=(segment_tag, [2] * len(segment_tag)),
            _class=_class
        )

        return self.augment_func(ret)


class DataProcess(DataHooks):
    data_dir: str
    seq_len: int

    val_dataset_ins = BaseDataset
    train_dataset_ins = BaseDataset

    train_data_num = None
    val_data_num = None

    vocab_size: int
    word_dict: dict
    vocab: set
    sp_tag_dict: dict

    is_mlm: bool
    is_nsp: bool

    def get_vocab(self):
        # note, vocab must be with word piece, e.g. uncased_L-12_H-768_A-12/vocab.txt
        # https://github.com/google-research/bert to get more details
        vocab = super().get_vocab()
        self.word_dict = {word: i for i, word in enumerate(vocab)}
        self.vocab = set(vocab)
        self.vocab_size = len(vocab)
        self.sp_tag_dict = {k: self.word_dict[v] for k, v in sp_token_dict.items()}
        self.sp_tag_dict.update(
            non_mask=-100,
            seg_pad=0
        )

    def _filter_func(self, x):
        if re.search('[0-9]', x):
            return False

        if re.search('[^a-z]', x):
            return False

        return True


class TextProcess(DataProcess):
    def make_vocab(self):
        # todo: make word piece
        iter_data = self.get_train_data()
        paragraphs = [ret['text'] for ret in iter_data]
        paragraphs = bundled.lower(paragraphs)
        segments = spliter.segments_from_paragraphs(paragraphs)
        word_dict = dict_maker.word_id_dict(segments, start_id=len(sp_token_dict), filter_func=self._filter_func)
        vocab = list(sp_token_dict.values()) + list(word_dict.keys())
        self.save_vocab(vocab)
        return vocab

    is_chunk = False

    def train_data_preprocess(self, iter_data, train=True):
        paragraphs = [ret['text'] for ret in iter_data]
        paragraphs = bundled.lower(paragraphs)
        segments = spliter.segments_from_paragraphs(paragraphs, is_word_piece=True, vocab=self.vocab, verbose=True)
        if self.is_chunk and train:
            segments = spliter.seg_chunks_from_segments(segments, max_length=self.seq_len - 2, min_length=self.seq_len / 8, verbose=True)

        iter_data = []
        for segment in segments:
            iter_data.append(dict(
                segment=segment,
                segment_tag=[1] * len(segment),
                text=' '.join(segment)
            ))
        return iter_data

    def val_data_preprocess(self, iter_data):
        return self.train_data_preprocess(iter_data, train=False)

    def count_seq_len(self):
        iter_data = self.get_train_data()
        iter_data = self.train_data_preprocess(iter_data)
        s = [len(ret['segment']) for ret in iter_data]
        self.log(f'mean seq len is {np.mean(s)}, max seq len is {np.max(s)}, min seq len is {np.min(s)}')

    def train_data_augment(self, ret, train=True) -> dict:
        ret = dict(ori_text=ret['text'])
        segments = [ret['segment']]
        segment_tags = [ret['segment_tag']]
        if train and self.is_mlm:
            segments, mask_tags = bundled.random_mask(segments, self.word_dict, mask_token=sp_token_dict['mask'], unk_tag=self.sp_tag_dict['unk'], non_mask_tag=self.sp_tag_dict['non_mask'])
            mask_tags = bundled.align(mask_tags, seq_len=self.seq_len, start_token=self.sp_tag_dict['non_mask'], end_token=self.sp_tag_dict['non_mask'], pad_token=self.sp_tag_dict['non_mask'])
            ret.update(mask_tag=mask_tags[0])

        segments = bundled.align(segments, seq_len=self.seq_len, start_token=sp_token_dict['cls'], end_token=sp_token_dict['sep'], pad_token=sp_token_dict['pad'])
        segment_tags = bundled.align(segment_tags, seq_len=self.seq_len, start_token=1, end_token=1, pad_token=self.sp_tag_dict['seg_pad'])
        text_tags = encoder.simple(segments, self.word_dict, unk_tag=self.sp_tag_dict['unk'])
        ret.update(
            segment=segments[0],
            text_tag=text_tags[0],
            segment_tag=segment_tags[0],
        )

        if self.is_nsp:
            ret.update(_class=ret['_class'])

        return ret

    def val_data_augment(self, ret) -> dict:
        return self.train_data_augment(ret)


class TextPairProcess(DataProcess):
    def make_vocab(self):
        # todo: make word piece
        iter_data = self.get_train_data()
        paragraphs = [' '.join(ret['texts']) for ret in iter_data]
        paragraphs = bundled.lower(paragraphs)
        segments = spliter.segments_from_paragraphs(paragraphs)
        word_dict = dict_maker.word_id_dict(segments, start_id=len(sp_token_dict), filter_func=self._filter_func)
        vocab = list(sp_token_dict.values()) + list(word_dict.keys())
        self.save_vocab(vocab)
        return vocab

    def train_data_preprocess(self, iter_data):
        text_pairs = [ret['texts'] for ret in iter_data]
        text_pairs = math_utils.transpose(text_pairs)
        tmp = []
        for paragraphs in text_pairs:
            paragraphs = bundled.lower(paragraphs)
            segments = spliter.segments_from_paragraphs(paragraphs, is_word_piece=True, vocab=self.vocab, verbose=True)
            tmp.append(segments)

        segment_pairs = math_utils.transpose(tmp)
        for ret, segment_pair in zip(iter_data, segment_pairs):
            # todo, replace unused token by unknown word here, and then save the vocab
            ret.update(
                segment_pair=segment_pair,
                segment_tag_pair=([1] * len(segment_pair[0]), [2] * len(segment_pair[1]))
            )

        return iter_data

    def val_data_preprocess(self, iter_data):
        return self.train_data_preprocess(iter_data)

    def count_seq_len(self):
        iter_data = self.get_train_data()
        iter_data = self.train_data_preprocess(iter_data)
        s = [len(ret['segment_pair'][0]) + len(ret['segment_pair'][1]) for ret in iter_data]
        self.log(f'mean seq len is {np.mean(s)}, max seq len is {np.max(s)}, min seq len is {np.min(s)}')

    def train_data_augment(self, ret, train=True) -> dict:
        """
        - dynamic mask(todo: add whole word mask)
        - add special token
        - encode(token id + segment id)
        """
        ret = dict(ori_texts=ret['texts'])
        segment_pairs = [ret['segment_pair']]
        segment_tag_pairs = [ret['segment_tag_pair']]

        if train and self.is_mlm:
            segment_pairs = math_utils.transpose(segment_pairs)
            tmp = []
            tmp2 = []
            for segments in segment_pairs:
                segments, mask_tags = bundled.random_mask(
                    segments, self.word_dict,
                    mask_token=sp_token_dict['mask'],
                    unk_tag=self.sp_tag_dict['unk'],
                    non_mask_tag=self.sp_tag_dict['non_mask']
                )
                tmp.append(segments)
                tmp2.append(mask_tags)

            segment_pairs = math_utils.transpose(tmp)

            mask_tags_pairs = math_utils.transpose(tmp2)
            mask_tags = bundled.joint(mask_tags_pairs, sep_token=self.sp_tag_dict['non_mask'])
            mask_tags = bundled.align(mask_tags, seq_len=self.seq_len, start_token=self.sp_tag_dict['non_mask'], pad_token=self.sp_tag_dict['non_mask'])
            ret.update(mask_tag=mask_tags[0])

        segments = bundled.joint(segment_pairs, sep_token=sp_token_dict['sep'])
        segments = bundled.align(segments, seq_len=self.seq_len, start_token=sp_token_dict['cls'], pad_token=sp_token_dict['pad'])

        segment_tags = bundled.joint(segment_tag_pairs, sep_token=1, keep_end=False)
        segment_tags = bundled.align(segment_tags, seq_len=self.seq_len, start_token=1, end_token=2, pad_token=self.sp_tag_dict['seg_pad'])

        text_tags = encoder.simple(segments, self.word_dict, unk_tag=self.sp_tag_dict['unk'])

        ret.update(
            segment=segments[0],
            text_tag=text_tags[0],
            segment_tag=segment_tags[0]
        )

        if self.is_nsp:
            ret.update(_class=ret['_class'])

        return ret

    def val_data_augment(self, ret) -> dict:
        return self.train_data_augment(ret, train=False)


class SimpleText(TextProcess):
    dataset_version = 'simple_text'
    data_dir: str
    seq_len = 64

    def get_train_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.SimpleText import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, return_label=self.is_nsp, generator=False)[0]

    def get_val_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.SimpleText import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.TEST, max_size=self.val_data_num, return_label=self.is_nsp, generator=False)[0]


class SimpleTextPair(TextPairProcess):
    dataset_version = 'simple_text_pair'
    data_dir: str
    seq_len = 64

    def get_train_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.SimpleTextPair import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]

    def get_val_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.SimpleTextPair import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.TEST, max_size=self.val_data_num, generator=False)[0]


class Bert(Process):
    model_version = 'bert'
    n_segment = 2
    is_mlm = True
    is_nsp = True

    def set_model(self):
        from models.text_pretrain.bert import Model

        self.get_vocab()
        self.model = Model(self.vocab_size, seq_len=self.seq_len, sp_tag_dict=self.sp_tag_dict, n_segment=self.n_segment)

    def set_optimizer(self):
        # todo, use the optimizer config from paper(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1), the training is failed
        # in RoBERTa, beta_2=0.98
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.5, 0.999))

    def on_train_step(self, rets, container, **kwargs) -> dict:
        mask_tags = torch.tensor([ret['mask_tag'] for ret in rets]).to(self.device)
        text_tags = torch.tensor([ret['text_tag'] for ret in rets]).to(self.device)
        segment_tags = torch.tensor([ret['segment_tag'] for ret in rets]).to(self.device)
        next_tags = torch.tensor([ret['_class'] for ret in rets]).to(self.device)

        output = self.model(text_tags, segment_tags, next_tags, mask_tags)

        return output

    def metric(self, *args, **kwargs) -> dict:
        from metrics import classification
        container = self.predict(**kwargs)

        metric_results = {}
        for name, results in container['model_results'].items():
            next_trues = np.array(results['next_trues'])
            next_preds = np.array(results['next_preds'])
            result = classification.pr.acc(next_trues, next_preds)

            mask_trues = np.array(results['mask_trues'])
            mask_preds = np.array(results['mask_preds'])
            # todo: quit pad token, the score will be more accurate
            score = np.sum(mask_trues == mask_preds) / mask_trues.size

            next_score = result['acc'],
            mask_score = score

            result.update(
                next_score=next_score,
                mask_score=mask_score,
                score=(next_score + mask_score) / 2
            )

            metric_results[name] = result

        return metric_results

    def on_val_step(self, rets, container, **kwargs) -> dict:
        text_tags = torch.tensor([ret['text_tag'] for ret in rets]).to(self.device)
        segment_tags = torch.tensor([ret['segment_tag'] for ret in rets]).to(self.device)

        models = container['models']
        model_results = {}
        for name, model in models.items():
            outputs = model(text_tags, segment_tags)

            model_results[name] = dict(
                next_outputs=outputs['next_pred'],
                next_preds=outputs['next_pred'].argmax(1).cpu().numpy().tolist(),
                mask_outputs=outputs['mask_pred'],
                mask_preds=outputs['mask_pred'].argmax(1).cpu().numpy().tolist()
            )

        return model_results

    def on_val_reprocess(self, rets, model_results, container, **kwargs):
        for name, results in model_results.items():
            r = container['model_results'].setdefault(name, dict())
            r.setdefault('next_trues', []).extend([ret['_class'] for ret in rets])
            r.setdefault('next_preds', []).extend(results['next_preds'])
            r.setdefault('mask_trues', []).extend([ret['text_tag'] for ret in rets])
            r.setdefault('mask_preds', []).extend(results['mask_preds'])
            r.setdefault('texts', []).extend([ret['ori_text'] for ret in rets])

    def on_val_end(self, container, is_visualize=False, **kwargs):
        if is_visualize:
            for name, results in container['model_results'].items():
                data = []
                for text, true, pred in zip(results['texts'], results['next_trues'], results['next_preds']):
                    d = dict(true=true, pred=pred)
                    if isinstance(text, str):
                        d['text'] = text
                    else:
                        d['text1'] = text[0]
                        d['text2'] = text[1]
                    data.append(d)
                df = pd.DataFrame(data)
                os_lib.Saver(stdout_method=self.log).auto_save(df, f'{self.cache_dir}/{self.counters["epoch"]}/{name}.csv', index=False)


class BertMLM(Bert):
    is_nsp = False

    def set_model(self):
        from models.text_pretrain.bert import Model

        self.get_vocab()
        self.model = Model(self.vocab_size, seq_len=self.seq_len, sp_tag_dict=self.sp_tag_dict, n_segment=self.n_segment, is_nsp=self.is_nsp)

    def on_train_step(self, rets, container, **kwargs) -> dict:
        mask_tags = torch.tensor([ret['mask_tag'] for ret in rets]).to(self.device)
        text_tags = torch.tensor([ret['text_tag'] for ret in rets]).to(self.device)
        segment_tags = torch.tensor([ret['segment_tag'] for ret in rets]).to(self.device)

        output = self.model(text_tags, segment_tags, mask_true=mask_tags)

        return output

    def metric(self, *args, **kwargs) -> dict:
        container = self.predict(**kwargs)

        metric_results = {}
        for name, results in container['model_results'].items():
            mask_trues = np.array(results['mask_trues'])
            mask_preds = np.array(results['mask_preds'])
            # todo: quit pad token, the score will be more accurate
            score = np.sum(mask_trues == mask_preds) / mask_trues.size
            result = dict(score=score)
            metric_results[name] = result

        return metric_results

    def on_val_step(self, rets, container, **kwargs) -> dict:
        text_tags = torch.tensor([ret['text_tag'] for ret in rets]).to(self.device)
        segment_tags = torch.tensor([ret['segment_tag'] for ret in rets]).to(self.device)

        models = container['models']
        model_results = {}
        for name, model in models.items():
            outputs = model(text_tags, segment_tags)

            model_results[name] = dict(
                mask_outputs=outputs['mask_pred'],
                mask_preds=outputs['mask_pred'].argmax(1).cpu().numpy().tolist()
            )

        return model_results

    def on_val_reprocess(self, rets, model_results, container, **kwargs):
        for name, results in model_results.items():
            r = container['model_results'].setdefault(name, dict())
            r.setdefault('mask_trues', []).extend([ret['text_tag'] for ret in rets])
            r.setdefault('mask_preds', []).extend(results['mask_preds'])
            r.setdefault('texts', []).extend([ret['ori_text'] for ret in rets])

    def on_val_end(self, container, is_visualize=False, **kwargs):
        if is_visualize:
            for name, results in container['model_results'].items():
                data = []
                for text, true, pred in zip(results['texts'], results['mask_trues'], results['mask_preds']):
                    d = dict(true=true, pred=pred)
                    if isinstance(text, str):
                        d['text'] = text
                    else:
                        d['text1'] = text[0]
                        d['text2'] = text[1]
                    data.append(d)
                df = pd.DataFrame(data)
                os_lib.Saver(stdout_method=self.log).auto_save(df, f'{self.cache_dir}/{self.counters["epoch"]}/{name}.csv', index=False)


class BertNSP(Bert):
    is_mlm = False

    def set_model(self):
        from models.text_pretrain.bert import Model

        self.get_vocab()
        self.model = Model(self.vocab_size, seq_len=self.seq_len, sp_tag_dict=self.sp_tag_dict, n_segment=self.n_segment, is_mlm=self.is_mlm)

    def on_train_step(self, rets, container, **kwargs) -> dict:
        text_tags = torch.tensor([ret['text_tag'] for ret in rets]).to(self.device)
        segment_tags = torch.tensor([ret['segment_tag'] for ret in rets]).to(self.device)
        next_tags = torch.tensor([ret['_class'] for ret in rets]).to(self.device)

        output = self.model(text_tags, segment_tags, next_true=next_tags)

        return output

    def metric(self, *args, **kwargs) -> dict:
        from metrics import classification
        container = self.predict(**kwargs)

        metric_results = {}
        for name, results in container['model_results'].items():
            next_trues = np.array(results['next_trues'])
            next_preds = np.array(results['next_preds'])
            result = classification.pr.acc(next_trues, next_preds)

            result.update(
                score=result['acc']
            )

            metric_results[name] = result

        return metric_results

    def on_val_step(self, rets, container, **kwargs) -> dict:
        text_tags = torch.tensor([ret['text_tag'] for ret in rets]).to(self.device)
        segment_tags = torch.tensor([ret['segment_tag'] for ret in rets]).to(self.device)

        models = container['models']
        model_results = {}
        for name, model in models.items():
            outputs = model(text_tags, segment_tags)

            model_results[name] = dict(
                next_outputs=outputs['next_pred'],
                next_preds=outputs['next_pred'].argmax(1).cpu().numpy().tolist(),
            )

        return model_results

    def on_val_reprocess(self, rets, model_results, container, **kwargs):
        for name, results in model_results.items():
            r = container['model_results'].setdefault(name, dict())
            r.setdefault('next_trues', []).extend([ret['_class'] for ret in rets])
            r.setdefault('next_preds', []).extend(results['next_preds'])
            r.setdefault('texts', []).extend([ret['ori_text'] for ret in rets])

    def on_val_end(self, container, is_visualize=False, **kwargs):
        if is_visualize:
            for name, results in container['model_results'].items():
                data = []
                for text, true, pred in zip(results['texts'], results['next_trues'], results['next_preds']):
                    d = dict(true=true, pred=pred)
                    if isinstance(text, str):
                        d['text'] = text
                    else:
                        d['text1'] = text[0]
                        d['text2'] = text[1]
                    data.append(d)
                df = pd.DataFrame(data)
                os_lib.Saver(stdout_method=self.log).auto_save(df, f'{self.cache_dir}/{self.counters["epoch"]}/{name}.csv', index=False)


class BertMLM_SimpleText(BertMLM, SimpleText):
    """
    Usage:
        .. code-block:: python

            from examples.text_pertain import BertMLM_SimpleText as Process

            Process().run(max_epoch=100, train_batch_size=128, check_period=3)
            {'score': 0.80395}      # about 200M data
    """
    is_chunk = True
