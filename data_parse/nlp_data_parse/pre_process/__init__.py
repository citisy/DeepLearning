import os
from utils import os_lib


class BertVocabOp:
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

    def __init__(self, vocab, word_dict=None, sp_token_dict=None, lower=False, non_mask_tag=-100, **kwargs):
        if lower:
            vocab = [v.lower() for v in vocab]

        self.vocab = set(vocab)
        self.vocab_size = len(vocab)
        self.word_dict = word_dict or {word: i for i, word in enumerate(vocab)}
        self.word_inv_dict = {v: k for k, v in self.word_dict.items()}
        self.sp_token_dict = sp_token_dict or self.sp_token_dict
        if lower:
            self.sp_token_dict = {k: v.lower() for k, v in self.sp_token_dict.items()}

        self.sp_tag_dict = {k: self.word_dict[v] for k, v in self.sp_token_dict.items() if v in self.word_dict}
        self.sp_tag_dict.update(non_mask=non_mask_tag)
        self.__dict__.update(kwargs)

    @staticmethod
    def from_pretrain(vocab_fn):
        # note, vocab must be with word piece, e.g. uncased_L-12_H-768_A-12/vocab.txt
        # https://github.com/google-research/bert to get more details
        vocab = os_lib.loader.auto_load(vocab_fn)
        return BertVocabOp(vocab)


class GptVocabOp:
    def __init__(self, byte_pairs, word_dict, pad_tag=None, **kwargs):
        import regex

        self.byte_pairs = byte_pairs
        self.word_dict = word_dict
        self.vocab_size = len(word_dict)
        self.pad_tag = pad_tag

        # https://github.com/openai/gpt-2/blob/master/src/encoder.py#L53
        self.sep_pattern = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.__dict__.update(kwargs)

    @staticmethod
    def from_pretrain(encoder_path, vocab_path):
        word_dict = os_lib.loader.load_json(encoder_path)
        byte_pairs = os_lib.loader.load_txt(vocab_path)
        byte_pairs = byte_pairs[1:]
        return GptVocabOp(byte_pairs, word_dict)
