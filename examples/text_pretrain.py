import torch
import numpy as np
from processor import Process, DataHooks, BaseDataset
from utils import math_utils, os_lib
from data_parse.nlp_data_parse.pre_process import spliter, encoder, bundled, sp_token_dict, dict_maker
import pandas as pd
import re


class RandomTextPairsDataset(BaseDataset):
    def __getitem__(self, idx):
        """all text pair in iter_data is the true text pair"""
        texts = self.iter_data[idx]['texts']
        _class = 1
        if np.random.random() < 0.5:  # 50% to select another text as the false sample
            tmp = np.random.choice(self.iter_data)['texts']
            texts = (texts[0], tmp[1])
            _class = 0

        ret = dict(
            texts=texts,
            _class=_class
        )

        return self.augment_func(ret)


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
        paragraphs = [' '.join(ret['texts']) for ret in iter_data]
        paragraphs = bundled.lower(paragraphs)
        segments = spliter.segments_from_paragraphs(paragraphs)
        word_dict = dict_maker.word_id_dict(segments, start_id=len(sp_token_dict), filter_func=filter_func)
        vocab = list(sp_token_dict.values()) + list(word_dict.keys())
        self.save_vocab(vocab)
        return vocab

    vocab_size: int
    word_dict: dict
    vocab: set
    sp_tag_dict: dict

    def get_vocab(self):
        # note, vocab must be with word piece, e.g. bert-base-uncased/vocab.txt
        vocab = super().get_vocab()
        self.word_dict = {word: i for i, word in enumerate(vocab)}
        self.vocab = set(vocab)
        self.vocab_size = len(vocab)
        self.sp_tag_dict = {k: self.word_dict[v] for k, v in sp_token_dict.items()}
        self.sp_tag_dict.update(
            non_mask=-100,
            seg_pad=0
        )

    def train_data_preprocess(self, iter_data):
        text_pairs = [ret['texts'] for ret in iter_data]
        text_pairs = math_utils.transpose(text_pairs)
        tmp = []
        for paragraphs in text_pairs:
            paragraphs = bundled.lower(paragraphs)
            segments = spliter.segments_from_paragraphs(paragraphs, is_word_piece=True, vocab=self.vocab)
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
        self.log(f'seq_len = {np.mean(s)}')


class Bert(Process):
    model_version = 'bert'

    def set_model(self):
        from models.text_pretrain.bert import Model

        self.get_vocab()
        self.model = Model(self.vocab_size, seq_len=self.seq_len, sp_tag_dict=self.sp_tag_dict, n_segment=2)

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
