import torch
import numpy as np
from processor import Process, DataHooks, BaseDataset
from utils import math_utils, os_lib
from data_parse.nlp_data_parse.pre_process import spliter, encoder, bundled, sp_token_dict, dict_maker
import pandas as pd
import re


class RandomTextPairsDataset(BaseDataset):
    def __getitem__(self, idx):
        """all text in iter_data is the true text pair"""
        texts = self.iter_data[idx]['texts']
        _class = 1
        if np.random.random() < 0.5:    # 50% to select another text as the false sample
            tmp = np.random.choice(self.iter_data)['texts']
            texts = (texts[0], tmp[1])
            _class = 0

        ret = dict(
            texts=texts,
            _class=_class
        )

        return self.augment_func(ret)


class QNLI(DataHooks):
    dataset_version = 'QNLI'

    val_dataset_ins = BaseDataset
    train_dataset_ins = BaseDataset

    train_data_num = None
    val_data_num = None

    data_dir = 'data/QNLI'
    vocab_fn = 'vocab.txt'  # note, vocab with word piece, e.g. bert-base-uncased/vocab.txt
    vocab_size: int
    word_dict: dict
    vocab: set
    sp_tag_dict: dict

    seq_len = 64

    def load_vocab(self):
        loader = os_lib.Loader(stdout_method=self.log)
        return loader.auto_load(f'{self.work_dir}/{self.vocab_fn}')

    def save_vocab(self, vocab):
        saver = os_lib.Saver(stdout_method=self.log)
        saver.auto_save(vocab, f'{self.work_dir}/{self.vocab_fn}')

    def make_vocab(self):
        # todo: make word piece
        def filter_func(x):
            if re.search('[0-9]', x):
                return False

            if re.search('[^a-z]', x):
                return False

            return True

        iter_data = self.get_train_data()
        paragraphs = [' '.join(ret['texts']) for ret in iter_data]
        paragraphs = bundled.lower(paragraphs)
        segments = spliter.segments_from_paragraphs(paragraphs)
        word_dict = dict_maker.word_id_dict(segments, start_id=len(sp_token_dict), filter_func=filter_func)
        vocab = list(sp_token_dict.values()) + list(word_dict.keys())
        self.save_vocab(vocab)
        return vocab

    def get_vocab(self):
        try:
            vocab = self.load_vocab()
        except OSError:
            vocab = self.make_vocab()

        self.word_dict = {word: i for i, word in enumerate(vocab)}
        self.vocab = set(vocab)
        self.vocab_size = len(vocab)
        self.sp_tag_dict = {k: self.word_dict[v] for k, v in sp_token_dict.items()}
        self.sp_tag_dict.update(
            non_mask=-100,
            seg_pad=2
        )

    def get_train_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.QNLI import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.TRAIN, max_size=self.train_data_num, generator=False)[0]

    def get_val_data(self, *args, **kwargs):
        from data_parse.nlp_data_parse.QNLI import Loader, DataRegister
        loader = Loader(self.data_dir)

        return loader.load(set_type=DataRegister.DEV, max_size=self.val_data_num, generator=False)[0]

    def train_data_augment(self, ret) -> dict:
        text_pairs = [ret['texts']]
        text_pairs = math_utils.transpose(text_pairs)
        tmp = []
        tmp2 = []
        for paragraphs in text_pairs:
            paragraphs = bundled.lower(paragraphs)
            segments = spliter.segments_from_paragraphs(paragraphs, is_word_piece=True, vocab=self.vocab)
            segments, mask_tags = bundled.random_mask(
                segments, self.word_dict,
                mask_token=sp_token_dict['mask'],
                unk_tag=self.sp_tag_dict['unk'],
                non_mask_tag=self.sp_tag_dict['non_mask']
            )
            tmp.append(segments)
            tmp2.append(mask_tags)

        text_pairs = math_utils.transpose(tmp)
        segments = bundled.joint(text_pairs, sep_token=sp_token_dict['sep'])
        segments = bundled.align(segments, seq_len=self.seq_len, start_token=sp_token_dict['cls'], pad_token=sp_token_dict['pad'])

        text_tags = encoder.simple(segments, self.word_dict, unk_tag=self.sp_tag_dict['unk'])
        segment_tags = encoder.seq_encode(segments, sep_token=sp_token_dict['sep'], max_sep=2)

        tag_pairs = math_utils.transpose(tmp2)
        mask_tags = bundled.joint(tag_pairs, sep_token=self.sp_tag_dict['non_mask'])
        mask_tags = bundled.align(mask_tags, seq_len=self.seq_len, start_token=self.sp_tag_dict['non_mask'], pad_token=self.sp_tag_dict['non_mask'])

        return dict(
            segment=segments[0],
            mask_tag=mask_tags[0],
            text_tag=text_tags[0],
            segment_tag=segment_tags[0],
            _class=ret['_class'],
            ori_texts=ret['texts']
        )

    def val_data_augment(self, ret) -> dict:
        text_pairs = [ret['texts']]
        text_pairs = math_utils.transpose(text_pairs)
        tmp = []
        for paragraphs in text_pairs:
            paragraphs = bundled.lower(paragraphs)
            segments = spliter.segments_from_paragraphs(paragraphs, is_word_piece=True, vocab=self.vocab)
            tmp.append(segments)

        text_pairs = math_utils.transpose(tmp)
        segments = bundled.joint(text_pairs, sep_token=sp_token_dict['sep'])
        segments = bundled.align(segments, seq_len=self.seq_len, start_token=sp_token_dict['cls'], pad_token=sp_token_dict['pad'])

        text_tags = encoder.simple(segments, self.word_dict, unk_tag=self.sp_tag_dict['unk'])
        segment_tags = encoder.seq_encode(segments, sep_token=sp_token_dict['sep'], max_sep=2)

        return dict(
            segment=segments[0],
            text_tag=text_tags[0],
            segment_tag=segment_tags[0],
            _class=ret['_class'],
            ori_texts=ret['texts']
        )


class Bert(Process):
    model_version = 'bert'

    def set_model(self):
        from models.text_pretrain.bert import Model

        self.get_vocab()
        self.model = Model(self.vocab_size, sp_tag_dict=self.sp_tag_dict)

    def set_optimizer(self):
        # todo, use the optimizer config from paper(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.1), the training is failed
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.5, 0.999))

    def set_stopper(self):
        from utils.torch_utils import EarlyStopping
        self.stopper = EarlyStopping(patience=10, min_epoch=10, ignore_min_score=0.6, stdout_method=self.log)

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
            trues = np.array(results['trues'])
            preds = np.array(results['preds'])
            result = classification.pr.acc(trues, preds)

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
                outputs=outputs['next_pred'],
                preds=outputs['next_pred'].argmax(1).cpu().numpy().tolist(),
            )

        return model_results

    def on_val_reprocess(self, rets, model_results, container, **kwargs):
        for name, results in model_results.items():
            r = container['model_results'].setdefault(name, dict())
            r.setdefault('trues', []).extend([ret['_class'] for ret in rets])
            r.setdefault('preds', []).extend(results['preds'])
            r.setdefault('texts', []).extend([ret['ori_texts'] for ret in rets])

    def on_val_end(self, container, is_visualize=False, **kwargs):
        if is_visualize:
            for name, results in container['model_results'].items():
                data = [dict(
                    true=true,
                    pred=pred,
                    text1=text[0],
                    text2=text[1],
                ) for text, true, pred in zip(results['texts'], results['trues'], results['preds'])]
                df = pd.DataFrame(data)
                os_lib.Saver(stdout_method=self.log).auto_save(df, f'{self.cache_dir}/{self.counters["epoch"]}/{name}.csv', index=False)


class Bert_QNLI(Bert, QNLI):
    """
    Usage:
        .. code-block:: python

            from examples.text_pair_classification import Bert_QNLI as Process

            Process().run(max_epoch=100, train_batch_size=128, predict_batch_size=128, check_period=3)
            {'score': 0.80395}
    """