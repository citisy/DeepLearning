import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from data_parse.nl_data_parse.pre_process import bundled, dict_maker, cleaner, snack
from processor import Process, DataHooks, BaseDataset, CheckpointHooks, IterIterDataset
from utils import math_utils, os_lib, torch_utils


class RandomChoiceTextPairsDataset(BaseDataset):
    """for Next Sentence Prediction"""

    def __getitem__(self, idx):
        """all text pair in iter_data is the true text pair"""
        ret = self.iter_data[idx]
        texts = ret['texts']
        segment_pair = ret['segment_pair']
        segment_pair_tags_pair = ret['segment_pair_tags_pair']

        # 50% to select another text as the false sample
        if np.random.random() < 0.5:
            next_ret = np.random.choice(self.iter_data)
            next_text = next_ret['texts'][1]
            next_segment = next_ret['segment_pair'][1]
            next_segment_pair_tags = next_ret['segment_pair_tags_pair'][1]

            texts = (texts[0], next_text)
            segment_pair = (segment_pair[0], next_segment)
            segment_pair_tags_pair = (segment_pair_tags_pair[0], next_segment_pair_tags)
            _class = 0

        else:
            _class = 1

        ret = dict(
            texts=texts,
            segment_pair=segment_pair,
            segment_pair_tags_pair=segment_pair_tags_pair,
            _class=_class
        )

        return self.augment_func(ret)


class RandomReverseTextPairsDataset(BaseDataset):
    """for Sentence Order Prediction"""

    def __getitem__(self, idx):
        ret = self.iter_data[idx]
        text = ret['text']
        segment = ret['segment']
        segment_pair_tags = ret['segment_pair_tags']

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
            segment_pair_tags_pair=(segment_pair_tags, [1] * len(segment_pair_tags)),
            _class=_class
        )

        return self.augment_func(ret)


class IterIterBatchDataset(IterIterDataset):
    """for loading large file"""

    def __iter__(self):
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()
        n = 1 if worker_info is None else worker_info.num_workers

        i = 0
        while i < self.length:
            rets = self.iter_data.get()
            for ret in self.process_batch(rets):
                yield ret

                i += n

    def process_batch(self, rets):
        if self.augment_func:
            rets = self.augment_func(rets)

        return rets


class DataProcessForBert(DataHooks):
    data_dir: str
    max_seq_len: int = 512

    val_dataset_ins = BaseDataset
    train_dataset_ins = BaseDataset

    train_data_num = None
    val_data_num = None

    is_token_cls: bool
    is_seq_cls: bool

    def _filter_func(self, x):
        if re.search('[0-9]', x):
            return False

        if re.search('[^a-z]', x):
            return False

        return True

    def save_vocab(self, vocab):
        saver = os_lib.Saver(stdout_method=self.log)
        saver.auto_save(vocab, f'{self.work_dir}/{self.vocab_fn}')


class TextProcessForBert(DataProcessForBert):
    def make_vocab(self):
        # todo: make word piece
        sp_token_dict = self.tokenizer.sp_token_dict
        iter_data = self.get_train_data()
        paragraphs = [ret['text'] for ret in iter_data]
        paragraphs = cleaner.Lower().from_paragraphs(paragraphs)
        segments = self.tokenizer.spliter.from_paragraphs(paragraphs)
        word_dict = dict_maker.word_id_dict(segments, start_id=len(sp_token_dict), filter_func=self._filter_func)
        vocab = list(sp_token_dict.values()) + list(word_dict.keys())
        self.save_vocab(vocab)
        return vocab

    is_chunk = False

    def data_preprocess(self, iter_data, train=True):
        paragraphs = [ret['text'] for ret in iter_data]
        # ['bert-base-uncased'] -> [['bert', '-', 'base', '-', 'un', '##cased']]
        segments = self.tokenizer.spliter.from_paragraphs(paragraphs)
        if not self.is_seq_cls and self.is_chunk and train:
            segments = self.tokenizer.chunker_spliter.from_segments(segments)

            iter_data = []
            for segment in segments:
                iter_data.append(dict(
                    segment=segment,
                    segment_pair_tags=[0] * len(segment),
                    text=' '.join(segment)
                ))
        else:
            for ret, segment in zip(iter_data, segments):
                # _class need
                ret.update(
                    segment=segment,
                    segment_pair_tags=[0] * len(segment),
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
        segment_pair_tags = [_ret['segment_pair_tags']]
        if train and self.is_token_cls:
            segments, mask_tags = self.tokenizer.perturbation.from_segments(segments)
            # while all([j == self.tokenizer.sp_id_dict['skip'] for i in mask_tags for j in i]):
            #     # to avoid nan loss
            #     segments, mask_tags = self.tokenizer.perturbation.from_segments(segments)
            ret.update(mask_tag=mask_tags[0])

        ret.update(
            segment=segments[0],
            segment_pair_tags=segment_pair_tags[0],
        )

        if self.is_seq_cls:
            ret.update(_class=_ret['_class'])

        return ret


class TextPairProcessForBert(DataProcessForBert):
    def make_vocab(self):
        # todo: make word piece
        sp_token_dict = self.tokenizer.sp_token_dict
        iter_data = self.get_train_data()
        paragraphs = [' '.join(ret['texts']) for ret in iter_data]
        paragraphs = cleaner.Lower().from_paragraphs(paragraphs)
        segments = self.tokenizer.spliter.from_paragraphs(paragraphs)
        word_dict = dict_maker.word_id_dict(segments, start_id=len(sp_token_dict), filter_func=self._filter_func)
        vocab = list(sp_token_dict.values()) + list(word_dict.keys())
        self.save_vocab(vocab)
        return vocab

    def data_preprocess(self, iter_data, train=True):
        text_pairs = [ret['texts'] for ret in iter_data]
        text_pairs = math_utils.transpose(text_pairs)
        tmp = []
        for paragraphs in text_pairs:
            segments = self.tokenizer.spliter.from_paragraphs(paragraphs)
            tmp.append(segments)

        segment_pairs = math_utils.transpose(tmp)
        for ret, segment_pair in zip(iter_data, segment_pairs):
            # todo, replace unused token by unknown word here, and then save the vocab
            ret.update(
                segment_pair=segment_pair,
                segment_pair_tags_pair=([0] * len(segment_pair[0]), [1] * len(segment_pair[1]))
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
        segment_pair_tags_pairs = [_ret['segment_pair_tags_pair']]

        if train and self.is_token_cls:
            segment_pairs = math_utils.transpose(segment_pairs)
            tmp = []
            tmp2 = []
            for segments in segment_pairs:
                segments, mask_tags = self.tokenizer.perturbation.from_segments(segments)
                tmp.append(segments)
                tmp2.append(mask_tags)

            segment_pairs = math_utils.transpose(tmp)

            mask_tags_pairs = math_utils.transpose(tmp2)
            mask_tags = snack.joint(mask_tags_pairs, sep_obj=self.tokenizer.skip_id, keep_end=False)
            ret.update(mask_tag=mask_tags[0])

        segments = snack.joint(segment_pairs, sep_obj=self.tokenizer.sep_token, keep_end=False)
        segment_pair_tags = snack.joint(segment_pair_tags_pairs, sep_obj=0, keep_end=False)

        ret.update(
            segment=segments[0],
            segment_pair_tags=segment_pair_tags[0]
        )

        if self.is_seq_cls:
            ret.update(_class=_ret['_class'])

        return ret


class SimpleTextForBert(TextProcessForBert):
    dataset_version = 'simple_text'
    data_dir: str

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nl_data_parse.datasets.SimpleText import Loader, DataRegister
        loader = Loader(self.data_dir)

        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, return_label=self.is_seq_cls, generator=False)[0]

        else:
            return loader.load(set_type=DataRegister.TEST, max_size=self.val_data_num, return_label=self.is_seq_cls, generator=False)[0]


class SimpleTextPairForBert(TextPairProcessForBert):
    dataset_version = 'simple_text_pair'
    data_dir: str

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nl_data_parse.datasets.SimpleTextPair import Loader, DataRegister
        loader = Loader(self.data_dir)

        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]
        else:
            return loader.load(set_type=DataRegister.TEST, max_size=self.val_data_num, generator=False)[0]


class LargeSimpleTextForBert(DataProcessForBert):
    """for loading large file"""
    dataset_version = 'simple_text'
    one_step_data_num = int(1e6)
    is_chunk = False

    def get_data(self, *args, train=True, batch_size=None, **kwargs):
        from data_parse.nl_data_parse.datasets.SimpleText import Loader, DataRegister
        import multiprocessing

        def gen_func():
            loader = Loader(self.data_dir)

            if train:
                iter_data = loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, return_label=self.is_seq_cls, generator=True)[0]
            else:
                iter_data = loader.load(set_type=DataRegister.TEST, max_size=self.val_data_num, return_label=self.is_seq_cls, generator=True)[0]

            rets = []
            for i, ret in enumerate(iter_data):
                rets.append(ret)
                if i % batch_size == batch_size - 1:
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

        if train:
            return IterIterBatchDataset(q, length=self.one_step_data_num, augment_func=self.train_data_augment)
        else:
            return IterIterBatchDataset(q, length=self.one_step_data_num, augment_func=self.val_data_augment)

    def data_augment(self, rets, train=True) -> List[dict]:
        """preprocess + data_augment"""
        rets = TextProcessForBert.data_preprocess(self, rets, train)
        rets = [TextProcessForBert.data_augment(self, ret, train) for ret in rets]
        return rets


class SOP(DataProcessForBert):
    train_dataset_ins = RandomReverseTextPairsDataset
    val_dataset_ins = RandomReverseTextPairsDataset

    dataset_version = 'simple_text'
    is_chunk = False

    def get_data(self, *args, train=True, **kwargs):
        from data_parse.nl_data_parse.datasets.SimpleText import Loader, DataRegister
        loader = Loader(self.data_dir)

        if train:
            return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, return_label=False, generator=False)[0]
        else:
            return loader.load(set_type=DataRegister.TEST, max_size=self.val_data_num, return_label=False, generator=False)[0]

    def make_vocab(self):
        return TextProcessForBert.make_vocab(self)

    def count_seq_len(self):
        return TextProcessForBert.count_seq_len(self)

    def data_preprocess(self, iter_data, train=True):
        return TextProcessForBert.data_preprocess(self, iter_data, train)

    def data_augment(self, ret, train=True) -> dict:
        return TextPairProcessForBert.data_augment(self, ret, train)


class BaseBert(Process):
    model_version = 'bert'
    is_token_cls = True
    is_seq_cls = True
    use_scaler = True
    scheduler_strategy = 'step'  # step
    max_seq_len: int

    def set_model(self):
        from models.text_pretrain.bert import Model
        self.model = Model(
            self.tokenizer.vocab_size,
            pad_id=self.tokenizer.pad_id,
            skip_id=self.tokenizer.skip_id,
            is_seq_cls=self.is_seq_cls, is_token_cls=self.is_token_cls
        )

    def set_tokenizer(self):
        self.tokenizer = bundled.BertTokenizer.from_pretrained(self.vocab_fn, max_seq_len=self.max_seq_len)

    def set_optimizer(self, lr=1e-4, betas=(0.5, 0.999), **kwargs):
        # todo, use the optimizer config from paper(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1), the training is failed
        # in RoBERTa, beta_2=0.98
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas)

    def get_model_inputs(self, loop_inputs, train=True):
        segments = [ret['segment'] for ret in loop_inputs]
        segment_pair_tags = [ret['segment_pair_tags'] for ret in loop_inputs]
        lens = [len(seg) for seg in segments]
        r = self.tokenizer.encode_segments(segments, segment_pair_tags)
        r = torch_utils.Converter.force_to_tensors(r, self.device)
        inputs = dict(
            x=r['segments_ids'],
            segment_label=r['segment_pair_tags'],
            attention_mask=r['valid_segment_tags'],
            lens=lens
        )

        if train:
            seq_cls_tags = torch.tensor([ret['_class'] for ret in loop_inputs]).to(self.device) if self.is_seq_cls else None
            token_cls_tags = None
            if self.is_token_cls:
                token_cls_tags = [ret['mask_tag'] for ret in loop_inputs]
                token_cls_tags = snack.align(
                    token_cls_tags, max_seq_len=self.max_seq_len,
                    start_obj=self.tokenizer.skip_id, end_obj=self.tokenizer.skip_id, pad_obj=self.tokenizer.skip_id
                )
                token_cls_tags = torch.tensor(token_cls_tags).to(self.device)

            inputs.update(
                seq_cls_true=seq_cls_tags,
                token_cls_true=token_cls_tags
            )

        return inputs

    def on_train_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        inputs = self.get_model_inputs(loop_inputs)
        with torch.cuda.amp.autocast(True):
            output = self.model(**inputs)

        return output

    def metric(self, *args, return_score='full', **kwargs) -> dict:
        """

        Args:
            *args:
            return_score: 'full', 'seq' or 'token'
            **kwargs:

        """
        from metrics import classification
        process_results = self.predict(**kwargs)

        metric_results = {}
        for name, results in process_results.items():
            result = {}
            if self.is_seq_cls:
                seq_cls_trues = np.array(results['seq_cls_trues'])
                seq_cls_preds = np.array(results['seq_cls_preds'])
                acc = np.sum(seq_cls_trues == seq_cls_preds) / seq_cls_trues.size
                seq_cls_result = classification.top_metric.f1(seq_cls_trues, seq_cls_preds)
                result.update({
                    'score.acc': acc,
                    'score.f1': seq_cls_result['f'],
                    **seq_cls_result})

            if self.is_token_cls:
                token_cls_trues = results['token_cls_trues']
                token_cls_preds = results['token_cls_preds']

                skip_id = self.tokenizer.skip_id
                token_cls_trues = snack.align(token_cls_trues, max_seq_len=self.max_seq_len, pad_obj=skip_id, pad_type=snack.MAX_LEN)
                token_cls_preds = snack.align(token_cls_preds, max_seq_len=self.max_seq_len, pad_obj=skip_id, pad_type=snack.MAX_LEN)

                token_cls_trues = np.array(token_cls_trues)
                token_cls_preds = np.array(token_cls_preds)
                n_quit = np.sum((token_cls_trues == skip_id) & (token_cls_preds == skip_id))
                n_true = np.sum(token_cls_trues == token_cls_preds)
                token_cls_score = (n_true - n_quit) / (token_cls_trues.size - n_quit)
                result.update({'score.token': token_cls_score})

            if return_score == 'seq':
                result.update(score=result['score.acc'])
            elif return_score == 'token':
                result.update(score=result['score.token'])
            elif return_score == 'full':
                if not self.is_seq_cls:
                    result.update(score=result['score.token'])
                elif not self.is_token_cls:
                    result.update(score=result['score.acc'])
                else:
                    result.update(score=(result['score.acc'] + result['score.token']) / 2)
            else:
                raise

            metric_results[name] = result

        return metric_results

    def on_val_step(self, loop_objs, **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)

        model_results = {}
        for name, model in self.models.items():
            outputs = model(**model_inputs)

            ret = dict()

            if self.is_seq_cls:
                ret.update(
                    seq_cls_logit=outputs['seq_cls_logit'],
                    seq_cls_preds=outputs['seq_cls_logit'].argmax(1).cpu().numpy().tolist(),
                )

            if self.is_token_cls:
                lens = model_inputs['lens']
                token_cls_preds = outputs['token_cls_logit'].argmax(-1).cpu().numpy().tolist()
                token_cls_preds = [preds[1: l + 1] for preds, l in zip(token_cls_preds, lens)]
                token_cls_trues = model_inputs['x'].cpu().numpy().tolist()
                token_cls_trues = [t[1: l + 1] for t, l in zip(token_cls_trues, lens)]
                ret.update(
                    token_cls_logit=outputs['token_cls_logit'],
                    token_cls_preds=token_cls_preds,
                    token_cls_trues=token_cls_trues,
                    pred_segment=self.tokenizer.numeralizer.decode(token_cls_preds),
                    true_segment=[ret['segment'] for ret in loop_inputs]
                )

            model_results[name] = ret

        return model_results

    def on_val_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        loop_inputs = loop_objs['loop_inputs']

        for name, results in model_results.items():
            r = process_results.setdefault(name, dict())
            r.setdefault('texts', []).extend([ret['ori_text'] for ret in loop_inputs])

            if self.is_seq_cls:
                r.setdefault('seq_cls_trues', []).extend([ret['_class'] for ret in loop_inputs])
                r.setdefault('seq_cls_preds', []).extend(results['seq_cls_preds'])

            if self.is_token_cls:
                r.setdefault('token_cls_trues', []).extend(results['token_cls_trues'])
                r.setdefault('token_cls_preds', []).extend(results['token_cls_preds'])

    def on_val_step_end(self, *args, **kwargs):
        """do not visualize"""

    def on_val_end(self, process_results=dict(), is_visualize=False, max_vis_num=None, epoch=-1, **kwargs):
        # todo: make a file to be submitted to https://gluebenchmark.com directly
        if is_visualize:
            for name, results in process_results.items():
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

                    if self.is_seq_cls:
                        d['seq_cls_true'] = results['seq_cls_trues'][i]
                        d['seq_cls_preds'] = results['seq_cls_preds'][i]

                    if self.is_token_cls:
                        d['token_cls_true'] = results['token_cls_trues'][i]
                        d['token_cls_preds'] = results['token_cls_preds'][i]

                    data.append(d)
                df = pd.DataFrame(data)
                os_lib.Saver(stdout_method=self.log).auto_save(df, f'{self.cache_dir}/{epoch}/{name}.csv', index=False)

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, **kwargs):
        rets = []
        for text in objs[0][start_idx: end_idx]:
            ret = dict(text=text)
            rets.append(ret)
        rets = self.val_data_preprocess(rets)
        return rets

    def on_predict_reprocess(self, loop_objs, process_results=dict(), **kwargs):
        model_results = loop_objs['model_results']
        for name, results in model_results.items():
            process_results.setdefault(name, []).extend(results['pred_segment'])


class FromBertHFPretrained(CheckpointHooks):
    """load pretrain model from hugging face"""

    def load_pretrained(self):
        if hasattr(self, 'pretrain_model'):
            from models.text_pretrain.bert import WeightLoader, WeightConverter
            state_dict = WeightLoader.from_hf(self.pretrain_model)
            state_dict = WeightConverter.from_hf(state_dict)
            self.model.load_state_dict(state_dict, strict=False)


class BertMLM(BaseBert, FromBertHFPretrained, TextProcessForBert):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pretrain import BertMLM as Process

            model_dir = 'xxx'
            process = Process(
                pretrain_model=f'{model_dir}/pytorch_model.bin',
                vocab_fn=f'{model_dir}/vocab.txt'
            )
            process.init()

            # if using `bert-base-uncased` pretrain model
            process.single_predict('The goal of life is [MASK].')
            # ['the', 'goal', 'of', 'life', 'is', 'life', '.']

            process.batch_predict([
                'The goal of life is [MASK].',
                'Paris is the [MASK] of France.'
            ])
            # ['the', 'goal', 'of', 'life', 'is', 'life', '.']
            # ['.', 'is', 'the', 'capital', 'of', 'france', '.']
    """
    dataset_version = ''
    is_seq_cls = False
    is_chunk = False


class BertMLM_SimpleText(BaseBert, SimpleTextForBert):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pretrain import BertMLM_SimpleText as Process

            Process(vocab_fn='...').run(max_epoch=20, train_batch_size=16, fit_kwargs=dict(check_period=1, accumulate=192))
    """
    is_seq_cls = False
    is_chunk = True


class Bert_SOP(BaseBert, SOP):
    """
    Usage:
        .. code-block:: python

            from bundles.text_pretrain import BertMLM_SimpleText as Process

            # about 200M data
            Process(vocab_fn='...').run(max_epoch=20, train_batch_size=16, fit_kwargs=dict(check_period=1, accumulate=192))
            {'score': 0.931447}
    """


class BgeM3(Process):
    """
    Usage:
        model_dir = 'xxx'
        processor = BgeM3(
            pretrain_model=model_dir,
            vocab_fn=f'{model_dir}/tokenizer.json',
            encoder_fn=f'{model_dir}/sentencepiece.bpe.model'
        )
        processor.init()

        text = [
            "What is BGE M3?",
            "Defination of BM25"
        ]
        outs = processor.batch_predict(text)
        dense_vecs = outs['dense_vecs']
        dense_vecs = torch.tensor(dense_vecs)
        similarity = dense_vecs[0][None] @ dense_vecs[1][None].T
        # [[0.4103]]

        outs = processor.batch_predict(
            text,
            model_kwargs=dict(
                return_sparse=True,
                return_colbert=True,
            )
        )
    """
    def set_model(self):
        from models.text_pretrain.bge_m3 import Model
        self.model = Model()

    def set_tokenizer(self):
        self.tokenizer = bundled.XLMRobertaTokenizer.from_pretrained(self.vocab_fn, self.encoder_fn)

    def load_pretrained(self):
        if self.pretrain_model:
            from models.text_pretrain.bge_m3 import WeightConverter
            from models.bundles import WeightLoader
            tensors = {
                'backbone': WeightLoader.auto_load(f'{self.pretrain_model}/pytorch_model.bin'),
                'sparse': WeightLoader.auto_load(f'{self.pretrain_model}/sparse_linear.pt'),
                'colbert': WeightLoader.auto_load(f'{self.pretrain_model}/colbert_linear.pt'),
            }
            tensors = WeightConverter.from_hf(tensors)
            self.model.load_state_dict(tensors, strict=False)

    def get_model_inputs(self, loop_inputs, train=True):
        paragraphs = [ret['text'] for ret in loop_inputs]
        r = self.tokenizer.encode_paragraphs(paragraphs)
        r = torch_utils.Converter.force_to_tensors(r, self.device)
        inputs = dict(
            input_ids=r['segments_ids'],
            attention_mask=r['valid_segment_tags'],
        )

        return inputs

    def on_val_step(self, loop_objs, model_kwargs=dict(), **kwargs) -> dict:
        loop_inputs = loop_objs['loop_inputs']
        model_inputs = self.get_model_inputs(loop_inputs, train=False)
        model_inputs.update(model_kwargs)

        model_results = {}
        for name, model in self.models.items():
            model_output = model(**model_inputs)
            model_results[name] = {k: v.cpu().numpy().tolist() for k, v in model_output.items()}

        return model_results

    def on_predict_reprocess(self, loop_objs, return_keys=(), **kwargs):
        super().on_predict_reprocess(
            loop_objs,
            return_keys=('dense_vecs', 'colbert_vecs', 'sparse_vecs'),
            **kwargs
        )

    def gen_predict_inputs(self, *objs, start_idx=None, end_idx=None, **kwargs) -> List[dict]:
        texts = objs[0]
        if isinstance(texts, str):
            texts = [texts] * (end_idx - start_idx)

        inputs = [dict(text=text) for text in texts]

        return inputs

