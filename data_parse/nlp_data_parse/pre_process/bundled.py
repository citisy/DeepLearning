import numpy as np
import torch
from . import spliter, chunker, cleaner, snack, numeralizer, perturbation
from utils import os_lib


class Apply:
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, obj):
        for func in self.funcs:
            obj = func(obj)
        return obj


class BertTokenizer:
    """the whole bert-like tokenize is like that:
    - input: 'hello world'
    - split: ['hello', 'world']
    - perturbation(non-essential): ['hello', '[MASK]']
    - add sp token: ['[CLS]', 'hello', '[MASK]', '[SEP]', '[PAD]', ...]
    - numerizer: [100, 1, 2, 101, 103, ...]
    """

    # base google vocab token
    unused_token_dict = {f'unused{i}': f'[unused{i}]' for i in range(1, 99)}
    sp_token_dict = dict(
        cls='[CLS]',
        sep='[SEP]',
        pad='[PAD]',
        unk='[UNK]',
        mask='[MASK]',
    )
    total_sp_token_dict = {
        **sp_token_dict,
        **unused_token_dict
    }

    def __init__(self, vocab, word_dict=None, sp_token_dict=None, skip_id=-100, max_seq_len=512, lower=False, **kwargs):
        self.max_seq_len = max_seq_len

        if lower:
            vocab = [v.lower() for v in vocab]

        self.vocab = set(vocab)
        self.vocab_size = len(vocab)
        self.word_dict = word_dict or {word: i for i, word in enumerate(vocab)}
        self.word_inv_dict = {v: k for k, v in self.word_dict.items()}
        self.sp_token_dict = sp_token_dict or self.sp_token_dict
        if lower:
            self.sp_token_dict = {k: v.lower() for k, v in self.sp_token_dict.items()}

        self.sp_id_dict = {k: self.word_dict[v] for k, v in self.sp_token_dict.items() if v in self.word_dict}
        self.sp_id_dict.update(skip=skip_id)

        self.__dict__.update(kwargs)
        self.__dict__.update({f'{k}_token': v for k, v in self.sp_token_dict.items()})
        self.__dict__.update({f'{k}_id': v for k, v in self.sp_id_dict.items()})

        self.spliter = spliter.ToSegments(
            cleaner=Apply(
                cleaner.Lower().from_paragraph,
                cleaner.StripAccents().from_paragraph,
            ),
            sp_tokens=set(self.sp_token_dict.values()),
            is_word_piece=True, vocab=self.vocab, verbose=True
        )
        self.chunker_spliter = chunker.ToChunkedSegments(
            max_length=self.max_seq_len - 2, min_length=self.max_seq_len / 8, verbose=True
        )
        self.numeralizer = numeralizer.KeyValueEncode(
            self.word_dict,
            self.word_inv_dict,
            unk_token=self.unk_token
        )
        self.perturbation = perturbation.RandomMask(self.word_dict, self.sp_id_dict, self.sp_token_dict)

    @staticmethod
    def from_pretrain(vocab_fn, **kwargs):
        # note, vocab must be with word piece, e.g. uncased_L-12_H-768_A-12/vocab.txt
        # https://github.com/google-research/bert to get more details
        vocab = os_lib.loader.auto_load(vocab_fn)
        return BertTokenizer(vocab, **kwargs)

    def encode(self, paragraphs, mask=False):
        segments = self.encode_paragraphs_to_segments(paragraphs)
        segment_pair_tags = [[0] * len(segment) for segment in segments]
        if mask:
            segments, mask_tags = self.perturbation.from_segments(segments)
        else:
            mask_tags = None

        r = self.encode_segments(segments, segment_pair_tags)

        return dict(
            **r,
            mask_tags=mask_tags
        )

    def decode(self):
        pass

    def encode_paragraphs_to_segments(self, paragraphs, is_chunk=False):
        segments = self.spliter.from_paragraphs(paragraphs)
        if is_chunk:
            segments = self.chunker_spliter.from_segments(segments)
        return segments

    def encode_segments(self, segments, segment_pair_tags):
        """

        Args:
            segments: [['hello', 'world', '[SEP]', 'hello', 'python'], ...]
            segment_pair_tags: [[0, 0, 0, 1, 1], ...]

        Returns:
            segments_ids: [[1, 2, 3, 4, 5, 0, 0, ...], ...]
            segment_pair_tags: [[0, 0, 0, 1, 1, 0, 0, ...], ...]
            valid_segment_tags: [[True, True, True, True, True, False, False, ...], ...]

        """
        valid_segment_tags = [[True] * len(seg) for seg in segments]
        seq_lens = [len(t) for t in segments]
        segments = snack.align(
            segments, max_seq_len=self.max_seq_len,
            start_obj=self.cls_token, end_obj=self.sep_token, pad_obj=self.pad_token
        )
        valid_segment_tags = snack.align(
            valid_segment_tags, max_seq_len=self.max_seq_len,
            start_obj=True, end_obj=True, pad_obj=False
        )
        segments_ids = self.numeralizer.encode(segments)
        segment_pair_tags = snack.align(
            segment_pair_tags, max_seq_len=self.max_seq_len,
            start_obj=0, end_obj=1, pad_obj=0
        )
        return dict(
            segments_ids=segments_ids,
            segment_pair_tags=segment_pair_tags,
            valid_segment_tags=valid_segment_tags,
            seq_lens=seq_lens
        )

    def decode_to_segments(self, segments_ids, valid_segment_tags):
        seq_lens = [sum(valid) for valid in valid_segment_tags]
        segments_ids = [seg[:seq_len] for seg, seq_len in zip(segments_ids, seq_lens)]
        segments = self.numeralizer.decode(segments_ids)
        return segments


class GPT2Tokenizer:
    """the whole bert-like tokenize is like that:
    - input: 'hello world'
    - split: ['hello', ' world']
    - numerizer(bpe): [1, 2]
    """

    sp_token_dict = dict(
        pad=' '
    )

    def __init__(self, byte_pairs, word_dict, pad_id=None, max_seq_len=512, **kwargs):
        import regex

        self.byte_pairs = byte_pairs
        self.word_dict = word_dict
        self.vocab_size = len(word_dict)
        self.max_seq_len = max_seq_len

        # https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
        self.sep_pattern = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self.spliter = spliter.ToSegments(sep_pattern=self.sep_pattern, is_split_punctuation=False)
        self.numerizer = numeralizer.BytePairEncode(self.byte_pairs, self.word_dict)
        self.pad_id = pad_id or self.numerizer.encode([' '])[0][0]

        self.__dict__.update(kwargs)

    @staticmethod
    def from_pretrain(encoder_path, vocab_path, **kwargs):
        word_dict = os_lib.loader.load_json(encoder_path)
        byte_pairs = os_lib.loader.load_txt(vocab_path)
        byte_pairs = byte_pairs[1:]
        return GPT2Tokenizer(byte_pairs, word_dict, **kwargs)

    def encode_segments(self, segments):
        segments_ids = self.numerizer.encode(segments)

        seq_lens = [len(t) for t in segments_ids]
        segments_ids = snack.align(segments_ids, max_seq_len=self.max_seq_len, pad_obj=self.pad_id)

        return dict(
            segments_ids=segments_ids,
            seq_lens=seq_lens
        )


class T5Tokenizer:
    additional_sp_token_dict = {f'extra_id_{i}': f'<extra_id_{i}>' for i in range(100)}
    sp_token_dict = dict(
        eos='</s>',
        pad='<pad>',
        unk='<unk>'
    )
    total_sp_token_dict = {
        **sp_token_dict,
        **additional_sp_token_dict
    }

    def __init__(self, sp_model: 'SentencePieceProcessor', max_seq_len=512, **kwargs):
        self.max_seq_len = max_seq_len
        self.sp_model = sp_model
        # note, there are 32000 in sp_model.vocab_size(), and 100 in additional_sp_token,
        # but got vocab_size of 32128 by official T5 model, so doubtful how to get the number
        # self.vocab_size: int = self.sp_model.vocab_size()
        self.vocab_size: int = 32128
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        self.__dict__.update(kwargs)
        self.__dict__.update({f'{k}_token': v for k, v in self.sp_token_dict.items()})

    @staticmethod
    def from_pretrain(vocab_path, **kwargs):
        # pip install sentencepiece
        from sentencepiece import SentencePieceProcessor

        sp_model = SentencePieceProcessor(model_file=vocab_path)
        return T5Tokenizer(sp_model, **kwargs)

    def encode(self, paragraphs):
        segments_ids = self.sp_model.encode(paragraphs)
        valid_segment_tags = [[True] * len(seg) for seg in segments_ids]
        seq_lens = [len(t) for t in segments_ids]
        segments_ids = snack.align(
            segments_ids, self.max_seq_len,
            end_obj=self.eos_id, pad_obj=self.pad_id
        )
        valid_segment_tags = snack.align(
            valid_segment_tags, max_seq_len=self.max_seq_len,
            end_obj=True, pad_obj=False
        )

        return dict(
            segments_ids=segments_ids,
            valid_segment_tags=valid_segment_tags,
            seq_lens=seq_lens
        )

    def decode(self, segments_ids):
        if isinstance(segments_ids, np.ndarray):
            segments_ids = segments_ids.tolist()
        elif isinstance(segments_ids, torch.Tensor):
            segments_ids = segments_ids.cpu().numpy().tolist()
        return self.sp_model.decode(segments_ids)


class CLIPTokenizer:
    sp_token_dict = dict(
        unk="<|endoftext|>",
        bos="<|startoftext|>",
        eos="<|endoftext|>",
        pad="<|endoftext|>",
    )

    def __init__(self, byte_pairs, word_dict, max_seq_len=77, **kwargs):
        import regex

        self.byte_pairs = byte_pairs
        self.word_dict = word_dict
        self.vocab_size = len(word_dict)
        self.max_seq_len = max_seq_len

        self.sep_pattern = regex.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE,
        )

        self.spliter = spliter.ToSegments(
            sep_pattern=self.sep_pattern,
            is_split_punctuation=False,
            cleaner=Apply(
                cleaner.Lower().from_paragraph,
            ),
            sp_tokens=set(self.sp_token_dict.values()),
        )
        self.numerizer = numeralizer.BytePairEncode(self.byte_pairs, self.word_dict)
        self.numerizer.make_chars = self.make_chars
        self.sp_id_dict = {k: self.word_dict.get(v) for k, v in self.sp_token_dict.items()}
        self.word_suffix = '</w>'

        self.__dict__.update(kwargs)
        self.__dict__.update({f'{k}_token': v for k, v in self.sp_token_dict.items()})
        self.__dict__.update({f'{k}_id': v for k, v in self.sp_id_dict.items()})

    def make_chars(self, word):
        """for bpe char pairs
        'hello' -> ['h', 'e', 'l', 'l', 'o</w>']"""
        return list(word[:-1]) + [word[-1] + self.word_suffix]

    @staticmethod
    def from_pretrain(encoder_path, vocab_path, **kwargs):
        word_dict = os_lib.loader.load_json(encoder_path)
        byte_pairs = os_lib.loader.load_txt(vocab_path)
        byte_pairs = byte_pairs[1:]
        return CLIPTokenizer(byte_pairs, word_dict, **kwargs)

    def encode_paragraphs(self, paragraphs):
        segments = self.spliter.from_paragraphs(paragraphs)
        r = self.encode_segments(segments)
        return r

    def encode_segments(self, segments):
        segments_ids = self.numerizer.encode(segments)

        seq_lens = [len(t) for t in segments_ids]
        segments_ids = snack.align(
            segments_ids, max_seq_len=self.max_seq_len, auto_pad=False,
            start_obj=self.bos_id,
            end_obj=self.eos_id,
            pad_obj=self.pad_id,
        )

        return dict(
            segments_ids=segments_ids,
            seq_lens=seq_lens
        )
