import torch
from processor import BaseDataset
from utils import math_utils, os_lib
from data_parse.nlp_data_parse.pre_process import spliter, encoder, bundled, sp_token_dict, dict_maker
from .text_pretrain import DataProcess, Bert as Process


class QNLI(DataProcess):
    dataset_version = 'QNLI'

    val_dataset_ins = BaseDataset
    train_dataset_ins = BaseDataset

    train_data_num = None
    val_data_num = None

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

    def train_data_augment(self, ret) -> dict:
        segment_pairs = [ret['segment_pair']]
        segment_tag_pairs = [ret['segment_tag_pair']]
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

        text_pairs = math_utils.transpose(tmp)
        segments = bundled.joint(text_pairs, sep_token=sp_token_dict['sep'])
        segments = bundled.align(segments, seq_len=self.seq_len, start_token=sp_token_dict['cls'], pad_token=sp_token_dict['pad'])

        segment_tags = bundled.joint(segment_tag_pairs, sep_token=1, keep_end=False)
        segment_tags = bundled.align(segment_tags, seq_len=self.seq_len, start_token=1, end_token=2, pad_token=self.sp_tag_dict['seg_pad'])

        mask_tags_pairs = math_utils.transpose(tmp2)
        mask_tags = bundled.joint(mask_tags_pairs, sep_token=self.sp_tag_dict['non_mask'])
        mask_tags = bundled.align(mask_tags, seq_len=self.seq_len, start_token=self.sp_tag_dict['non_mask'], pad_token=self.sp_tag_dict['non_mask'])

        text_tags = encoder.simple(segments, self.word_dict, unk_tag=self.sp_tag_dict['unk'])

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
    train_pretrain = False
    pretrain_model: str

    def set_model(self):
        self.get_vocab()

        if self.train_pretrain:
            from models.text_pretrain.bert import Model

            self.model = Model(self.vocab_size, seq_len=self.seq_len, sp_tag_dict=self.sp_tag_dict, n_segment=2)

        else:
            from models.text_pair_classification.bert import Model

            self.model = Model(self.vocab_size, seq_len=self.seq_len, sp_tag_dict=self.sp_tag_dict)
            ckpt = torch.load(self.pretrain_model, map_location=self.device)
            self.model.load_state_dict(ckpt['model'], strict=False)


class Bert_QNLI(Bert, QNLI):
    """
    Usage:
        .. code-block:: python

            from examples.text_pair_classification import Bert_QNLI as Process

            Process(train_pretrain=True).run(max_epoch=100, train_batch_size=128, predict_batch_size=128, check_period=3)
            {'score': 0.80395}  # no pretrain data, use QNLI data to train directly
    """
