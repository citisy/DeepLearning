import os
import torch
import numpy as np
from processor import Process, DataHooks, BaseDataset, ModelHooks, CheckpointHooks, IterBatchDataset
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
        segment_pair = ret['segment_pair']
        segment_tag_pair = ret['segment_tag_pair']

        # 50% to select another text as the false sample
        if np.random.random() < 0.5:
            next_ret = np.random.choice(self.iter_data)
            next_text = next_ret['texts'][1]
            next_segment = next_ret['segment_pair'][1]
            next_segment_tag = next_ret['segment_tag_pair'][1]

            texts = (texts[0], next_text)
            segment_pair = (segment_pair[0], next_segment)
            segment_tag_pair = (segment_tag_pair[0], next_segment_tag)
            _class = 0

        else:
            _class = 1

        ret = dict(
            texts=texts,
            segment_pair=segment_pair,
            segment_tag_pair=segment_tag_pair,
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
            segment_tag_pair=(segment_tag, [1] * len(segment_tag)),
            _class=_class
        )

        return self.augment_func(ret)


class DataProcess(DataHooks):
    data_dir: str
    max_seq_len: int

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
        self.sp_tag_dict.update(non_mask=-100)

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

    def data_preprocess(self, iter_data, train=True):
        paragraphs = [ret['text'] for ret in iter_data]
        paragraphs = bundled.lower(paragraphs)
        segments = spliter.segments_from_paragraphs(paragraphs, is_word_piece=True, vocab=self.vocab, verbose=True)
        if not self.is_nsp and self.is_chunk and train:
            segments = spliter.chunked_segments_from_segments(segments, max_length=self.max_seq_len - 2, min_length=self.max_seq_len / 8, verbose=True)

            iter_data = []
            for segment in segments:
                iter_data.append(dict(
                    segment=segment,
                    segment_tag=[0] * len(segment),
                    text=' '.join(segment)
                ))
        else:
            for ret, segment in zip(iter_data, segments):
                ret.update(
                    segment=segment,
                    segment_tag=[0] * len(segment),
                    text=' '.join(segment)
                )
        return iter_data

    def count_seq_len(self):
        iter_data = self.get_train_data()
        iter_data = self.train_data_preprocess(iter_data)
        s = [len(ret['segment']) for ret in iter_data]
        self.log(f'mean seq len is {np.mean(s)}, max seq len is {np.max(s)}, min seq len is {np.min(s)}')

    def data_augment(self, ret, train=True) -> dict:
        _ret = ret
        ret = dict(ori_text=_ret['text'])
        segments = [_ret['segment']]
        segment_tags = [_ret['segment_tag']]
        if train and self.is_mlm:
            segments, mask_tags = bundled.random_mask(segments, self.word_dict, mask_token=sp_token_dict['mask'], unk_tag=self.sp_tag_dict['unk'], non_mask_tag=self.sp_tag_dict['non_mask'])
            ret.update(mask_tag=mask_tags[0])

        ret.update(
            segment=segments[0],
            segment_tag=segment_tags[0],
        )

        if self.is_nsp:
            ret.update(_class=_ret['_class'])

        return ret


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

    def data_preprocess(self, iter_data, train=True):
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
                segment_tag_pair=([0] * len(segment_pair[0]), [1] * len(segment_pair[1]))
            )

        return iter_data

    def count_seq_len(self):
        iter_data = self.get_train_data()
        iter_data = self.train_data_preprocess(iter_data)
        s = [len(ret['segment_pair'][0]) + len(ret['segment_pair'][1]) for ret in iter_data]
        self.log(f'mean seq len is {np.mean(s)}, max seq len is {np.max(s)}, min seq len is {np.min(s)}')

    def data_augment(self, ret, train=True) -> dict:
        """
        - dynamic mask(todo: add whole word mask)
        - add special token
        - encode(token id + segment id)
        """
        _ret = ret
        ret = dict(ori_text=_ret['texts'])
        segment_pairs = [_ret['segment_pair']]
        segment_tag_pairs = [_ret['segment_tag_pair']]

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
            mask_tags = bundled.joint(mask_tags_pairs, sep_token=self.sp_tag_dict['non_mask'], keep_end=False)
            ret.update(mask_tag=mask_tags[0])

        segments = bundled.joint(segment_pairs, sep_token=sp_token_dict['sep'], keep_end=False)
        segment_tags = bundled.joint(segment_tag_pairs, sep_token=0, keep_end=False)

        ret.update(
            segment=segments[0],
            segment_tag=segment_tags[0]
        )

        if self.is_nsp:
            ret.update(_class=_ret['_class'])

        return ret


class SimpleText(TextProcess):
    dataset_version = 'simple_text'
    data_dir: str
    max_seq_len = 512

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nlp_data_parse.SimpleText import Loader, DataRegister
        loader = Loader(self.data_dir)

        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, return_label=self.is_nsp, generator=False)[0]

        else:
            return loader.load(set_type=DataRegister.TEST, max_size=self.val_data_num, return_label=self.is_nsp, generator=False)[0]


class SimpleTextPair(TextPairProcess):
    dataset_version = 'simple_text_pair'
    data_dir: str
    max_seq_len = 512

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nlp_data_parse.SimpleTextPair import Loader, DataRegister
        loader = Loader(self.data_dir)

        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]
        else:
            return loader.load(set_type=DataRegister.TEST, max_size=self.val_data_num, generator=False)[0]


class IterBatchSimpleText(TextProcess):
    """for loading large file"""
    dataset_version = 'simple_text'
    data_dir: str
    max_seq_len = 512
    iter_batch_num = 1000
    one_step_data_num = int(1e6)

    train_dataset_ins = IterBatchDataset
    val_dataset_ins = IterBatchDataset

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nlp_data_parse.SimpleText import Loader, DataRegister
        import multiprocessing

        self.train_dataset_ins.length = self.one_step_data_num
        self.val_dataset_ins.length = self.one_step_data_num

        def gen_func():
            loader = Loader(self.data_dir)

            if train:
                iter_data = loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, return_label=self.is_nsp, generator=True)[0]

            else:
                iter_data = loader.load(set_type=DataRegister.TEST, max_size=self.val_data_num, return_label=self.is_nsp, generator=True)[0]

            rets = []
            for i, ret in enumerate(iter_data):
                rets.append(ret)
                if i % self.iter_batch_num == self.iter_batch_num - 1:
                    yield rets
                    rets = []

                if rets:
                    yield rets

        def producer(q):
            iter_data = gen_func()
            while True:
                if not q.full():
                    try:
                        q.put(next(iter_data))
                    except StopIteration:
                        iter_data = gen_func()
                        q.put(next(iter_data))

        q = multiprocessing.Queue(8)
        p = multiprocessing.Process(target=producer, args=(q,))
        p.daemon = True
        p.start()

        return q


class SOP(DataProcess):
    train_dataset_ins = RandomReverseTextPairsDataset
    val_dataset_ins = RandomReverseTextPairsDataset

    dataset_version = 'simple_text'
    data_dir: str
    max_seq_len = 512

    is_chunk = False

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nlp_data_parse.SimpleText import Loader, DataRegister
        loader = Loader(self.data_dir)

        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, return_label=False, generator=False)[0]
        else:
            return loader.load(set_type=DataRegister.TEST, max_size=self.val_data_num, return_label=False, generator=False)[0]

    def make_vocab(self):
        return TextProcess.make_vocab(self)

    def count_seq_len(self):
        return TextProcess.count_seq_len(self)

    def data_preprocess(self, iter_data, train=True):
        return TextProcess.data_preprocess(self, iter_data, train)

    def data_augment(self, ret, train=True) -> dict:
        return TextPairProcess.data_augment(self, ret, train)


class Bert(Process):
    model_version = 'bert'
    is_mlm = True
    is_nsp = True
    use_scaler = True

    def set_model(self):
        from models.text_pretrain.bert import Model

        self.get_vocab()
        self.model = Model(
            self.vocab_size,
            sp_tag_dict=self.sp_tag_dict,
            is_nsp=self.is_nsp, is_mlm=self.is_mlm
        )

    def set_optimizer(self):
        # todo, use the optimizer config from paper(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1), the training is failed
        # in RoBERTa, beta_2=0.98
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.5, 0.999))

    def set_scheduler(self, max_epoch):
        if self.use_scheduler:
            from transformers import get_scheduler

            self.scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=max_epoch * len(self.train_container['train_dataloader']),
            )
            self.scheduler_strategy = 1  # step

    def get_model_inputs(self, rets, train=True):
        segments = [ret['segment'] for ret in rets]
        attention_mask = [[True] * len(seg) for seg in segments]
        segments = bundled.align(segments, max_seq_len=self.max_seq_len, start_token=sp_token_dict['cls'], end_token=sp_token_dict['sep'], pad_token=sp_token_dict['pad'])
        attention_mask = bundled.align(attention_mask, max_seq_len=self.max_seq_len, start_token=True, end_token=True, pad_token=False)
        token_tags = encoder.simple(segments, self.word_dict, unk_tag=self.sp_tag_dict['unk'])

        segment_tags = [ret['segment_tag'] for ret in rets]
        segment_tags = bundled.align(segment_tags, max_seq_len=self.max_seq_len, start_token=0, end_token=1, pad_token=0)

        token_tags = torch.tensor(token_tags).to(self.device)
        segment_tags = torch.tensor(segment_tags).to(self.device)
        attention_mask = torch.tensor(attention_mask).to(self.device)

        r = dict(
            x=token_tags,
            segment_label=segment_tags,
            attention_mask=attention_mask,
        )

        if train:
            next_tags = torch.tensor([ret['_class'] for ret in rets]).to(self.device) if self.is_nsp else None
            mask_tags = None
            if self.is_mlm:
                mask_tags = [ret['mask_tag'] for ret in rets]
                mask_tags = bundled.align(mask_tags, max_seq_len=self.max_seq_len, start_token=self.sp_tag_dict['non_mask'], end_token=self.sp_tag_dict['non_mask'], pad_token=self.sp_tag_dict['non_mask'])
                mask_tags = torch.tensor(mask_tags).to(self.device)

            r.update(
                next_true=next_tags,
                mask_true=mask_tags
            )

        return r

    def on_train_step(self, rets, **kwargs) -> dict:
        rets = self.get_model_inputs(rets)
        with torch.cuda.amp.autocast(True):
            output = self.model(**rets)

        return output

    def metric(self, *args, return_score='full', **kwargs) -> dict:
        """

        Args:
            *args:
            return_score: 'full', 'next' or 'mask'
            **kwargs:

        """
        from metrics import classification
        container = self.predict(**kwargs)

        metric_results = {}
        for name, results in container['model_results'].items():
            result = {}
            if self.is_nsp:
                next_trues = np.array(results['next_trues'])
                next_preds = np.array(results['next_preds'])
                acc = np.sum(next_trues == next_preds) / next_trues.size
                next_result = classification.top_metric.f1(next_trues, next_preds)
                result.update({
                    'score.acc': acc,
                    'score.f1': next_result['f'],
                    **next_result})

            if self.is_mlm:
                mask_trues = results['mask_trues']
                mask_preds = results['mask_preds']

                mask_trues = bundled.align(mask_trues, max_seq_len=self.max_seq_len, start_token=self.sp_tag_dict['non_mask'], pad_token=self.sp_tag_dict['non_mask'])
                mask_preds = bundled.align(mask_preds, max_seq_len=self.max_seq_len, start_token=self.sp_tag_dict['non_mask'], pad_token=self.sp_tag_dict['non_mask'])

                mask_trues = np.array(mask_trues)
                mask_preds = np.array(mask_preds)
                # todo: quit pad token, the score will be more accurate
                mask_score = np.sum(mask_trues == mask_preds) / mask_trues.size
                result.update({'score.mask': mask_score})

            if return_score == 'next':
                result.update(score=result['score.acc'])
            elif return_score == 'mask':
                result.update(score=result['score.mask'])
            elif return_score == 'full':
                if not self.is_nsp:
                    result.update(score=result['score.mask'])
                elif not self.is_mlm:
                    result.update(score=result['score.acc'])
                else:
                    result.update(score=(result['score.acc'] + result['score.mask']) / 2)
            else:
                raise

            metric_results[name] = result

        return metric_results

    def on_val_step(self, rets, **kwargs) -> dict:
        rets = self.get_model_inputs(rets, train=False)
        token_tags = rets['x']

        model_results = {}
        for name, model in self.models.items():
            outputs = model(**rets)

            ret = dict()

            if self.is_nsp:
                ret.update(
                    next_outputs=outputs['next_pred'],
                    next_preds=outputs['next_pred'].argmax(1).cpu().numpy().tolist(),
                )

            if self.is_mlm:
                ret.update(
                    mask_outputs=outputs['mask_pred'],
                    mask_preds=outputs['mask_pred'].argmax(1).cpu().numpy().tolist(),
                    token_tags=token_tags.cpu().numpy().tolist()
                )

            model_results[name] = ret

        return model_results

    def on_val_reprocess(self, rets, model_results, **kwargs):
        for name, results in model_results.items():
            r = self.val_container['model_results'].setdefault(name, dict())
            r.setdefault('texts', []).extend([ret['ori_text'] for ret in rets])

            if self.is_nsp:
                r.setdefault('next_trues', []).extend([ret['_class'] for ret in rets])
                r.setdefault('next_preds', []).extend(results['next_preds'])

            if self.is_mlm:
                r.setdefault('mask_trues', []).extend(results['token_tags'])
                r.setdefault('mask_preds', []).extend(results['mask_preds'])

    def on_val_step_end(self, *args, **kwargs):
        pass

    def on_val_end(self, is_visualize=False, max_vis_num=None, **kwargs):
        # todo: make a file to be submitted to https://gluebenchmark.com directly
        if is_visualize:
            for name, results in self.val_container['model_results'].items():
                data = []
                vis_num = max_vis_num or len(results['texts'])
                for i in range(vis_num):
                    text = results['texts'][i]
                    d = dict()
                    if isinstance(text, str):
                        d['text'] = text
                    else:
                        d['text1'] = text[0]
                        d['text2'] = text[1]

                    if self.is_nsp:
                        d['next_true'] = results['next_trues'][i]
                        d['next_pred'] = results['next_preds'][i]

                    if self.is_mlm:
                        d['mask_true'] = results['mask_trues'][i]
                        d['mask_pred'] = results['mask_preds'][i]

                    data.append(d)
                df = pd.DataFrame(data)
                os_lib.Saver(stdout_method=self.log).auto_save(df, f'{self.cache_dir}/{self.counters["epoch"]}/{name}.csv', index=False)


class FromHFPretrain(CheckpointHooks):
    """load pretrain model from hugging face"""

    def load_pretrain(self):
        if hasattr(self, 'pretrain_model'):
            from models.text_pretrain.bert import convert_hf_weights

            if os.path.exists(f'{self.pretrain_model}/pytorch_model.bin'):
                state_dict = torch.load(f'{self.pretrain_model}/pytorch_model.bin', map_location=self.device)
            else:  # download weight auto
                from transformers import BertForSequenceClassification
                model = BertForSequenceClassification.from_pretrained(self.pretrain_model, num_labels=2)
                state_dict = model.state_dict()

            self.model.load_state_dict(convert_hf_weights(state_dict), strict=False)


class BertMLM_SimpleText(Bert, SimpleText):
    """
    Usage:
        .. code-block:: python

            from examples.text_pertain import BertMLM_SimpleText as Process

            Process(device=0).run(max_epoch=20, train_batch_size=16, fit_kwargs=dict(check_period=1, accumulate=192))
    """
    is_nsp = False
    is_chunk = True


class Bert_SOP(Bert, SOP):
    """
    Usage:
        .. code-block:: python

            from examples.text_pertain import BertMLM_SimpleText as Process

            # about 200M data
            Process(device=0).run(max_epoch=20, train_batch_size=16, fit_kwargs=dict(check_period=1, accumulate=192))
            {'score': 0.931447}
    """
