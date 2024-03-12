import copy
import numpy as np
from typing import List
from . import BertVocabOp
from utils import math_utils


class JitterBase:
    def __init__(
            self, word_dict, sp_tag_dict, sp_token_dict=BertVocabOp.total_sp_token_dict,
            include_tokens=None, exclude_tokens=(),
            prob=0.15, **kwargs
    ):
        self.word_dict = word_dict

        self.sp_token_dict = sp_token_dict
        self.sp_tag_dict = sp_tag_dict
        self.mask_token = self.sp_token_dict['mask']
        self.unk_tag = self.sp_tag_dict['unk']
        self.non_mask_tag = self.sp_tag_dict['non_mask']

        self.exclude_tokens = set(exclude_tokens) | set(sp_token_dict.values())
        self.include_tokens = set(include_tokens or word_dict.keys())
        self.vocab = list(self.include_tokens - self.exclude_tokens)

        self.prob = prob
        self.__dict__.update(kwargs)

    def from_segments(self, segments: List[List[str]], **kwargs):
        rets = [self.from_segment(segment, **kwargs) for segment in segments]
        return math_utils.transpose(rets)

    def from_segment(self, segment: List[str], **kwargs):
        segment = copy.deepcopy(segment)
        probs = np.random.uniform(0., 1., len(segment))
        mask_tag = np.full_like(probs, self.non_mask_tag, dtype=int).tolist()

        shift = 0
        for i in np.where(probs < self.prob)[0]:
            _, shift = self.segment_inplace(segment, mask_tag, i, shift, **kwargs)

        return segment, mask_tag

    def segment_inplace(self, segment, mask_tag, i, shift, **kwargs):
        """return:
            j: where has been replaced
            shift: j = i + shift, where will be replaced
        """
        raise NotImplemented

    def make_one_token(self, segment, j, **kwargs):
        return np.random.choice(self.vocab)

    def make_many_tokens(self, segment, j, n_tokens=2, **kwargs):
        return np.random.choice(self.vocab, n_tokens)

    def skip(self, segment, mask_tag, j, **kwargs):
        """true to skip"""
        return (
                (j >= len(segment) and j >= len(mask_tag))
                or mask_tag[j] != self.non_mask_tag  # if the position of token has been change, skip
                or segment[j] in self.exclude_tokens or segment[j] not in self.include_tokens
        )


class TokenJitter(JitterBase):
    """
    Usage:
        rand = perturbation.TokenJitter([
            perturbation.RandomReplace(out_vocab_op.word_dict, out_vocab_op.sp_tag_dict),
            perturbation.RandomAppend(out_vocab_op.word_dict, out_vocab_op.sp_tag_dict),
            perturbation.RandomDelete(out_vocab_op.word_dict, out_vocab_op.sp_tag_dict),
        ])
        segments = rand.from_segments(segments)
    """

    def __init__(self, ins: List[JitterBase], prob=0.15, ins_probs=None, **kwargs):
        self.ins = ins
        self.ins_probs = ins_probs
        self.prob = prob
        self.non_mask_tag = self.ins[0].non_mask_tag

    def skip(self, segment, mask_tag, j, choice_idx=None, **kwargs):
        return False

    def segment_inplace(self, segment, mask_tag, i, shift, **kwargs):
        choice_idx = np.random.choice(len(self.ins), p=self.ins_probs)

        if self.skip(segment, mask_tag, i, choice_idx=choice_idx):
            return None, shift

        ins = self.ins[choice_idx]
        j, shift = ins.segment_inplace(segment, mask_tag, i, shift)
        if j is not None:
            js = j if isinstance(j, list) else [j]
            for j in js:
                mask_tag[j] = (choice_idx, mask_tag[j])

        return j, shift


class RandomMask(JitterBase):
    """
    ['hello', 'world', 'hello', 'python']
    -> segment = ['hello', '[MASK]', 'hello', 'java']
    -> mask_tag = [-100, 1, -100, 2]

    -100 is non_mask_tag, means the token in the position is not masked,
    1 and 2 are tags of 'world' and 'python', means the token got masked
    """

    mask_prob = 0.8
    replace_prob = 0.1

    def segment_inplace(self, segment, mask_tag, i, shift, **kwargs):
        j = i + shift
        token = segment[j]
        if self.skip(segment, mask_tag, j):
            return None, shift

        prob = np.random.random()
        # mask_prob to add [MASK] token
        if prob < self.mask_prob:
            segment[j] = self.mask_token

        # replace_prob to change to another word
        elif prob < self.mask_prob + self.replace_prob:
            segment[j] = self.make_one_token(segment, j)

        mask_tag[j] = self.word_dict.get(token, self.unk_tag)
        return j, shift


class RandomReplace(JitterBase):
    """
    ['hello', 'world', 'hello', 'python']
    -> segment = ['hello', 'word', 'hello', 'java']
    -> mask_tag = [-100, 1, -100, 2]

    -100 is non_mask_tag, means the token in the position is not replaced,
    1 and 2 are tags of 'world' and 'python', means the token got replaced
    """

    def segment_inplace(self, segment, mask_tag, i, shift, **kwargs):
        j = i + shift

        if self.skip(segment, mask_tag, j):
            return None, shift

        token = segment[j]
        segment[j] = self.make_one_token(segment, j)
        mask_tag[j] = self.word_dict.get(token, self.unk_tag)
        return j, shift


class RandomReplaceTokens(RandomReplace):
    n_tokens = 2

    def segment_inplace(self, segment, mask_tag, i, shift, **kwargs):
        j = i + shift

        for a in range(self.n_tokens):
            if self.skip(segment, mask_tag, j + a):
                return None, shift

        change_tokens = self.make_many_tokens(segment, j, n_tokens=self.n_tokens)

        js = []
        for change_token in change_tokens:
            js.append(j)
            token = segment[j]
            segment[j] = change_token
            mask_tag[j] = self.word_dict.get(token, self.unk_tag)
            j += 1
        return js, shift


class RandomAppend(JitterBase):
    """
    ['hello', 'hello']
    if keep_len = True:
        -> segment = ['hello', 'hello']
        -> mask_tag = [1, 2]

        -100 is non_mask_tag, means the token in the position is not appended,
        1 and 2 is tag of 'world' and 'python', means one token will be got appended after

    if keep_len = False:
        -> segment = ['hello', 'world', 'hello', 'python']
        -> mask_tag = [-100, 1, -100, 2]

        -100 is non_mask_tag, means the token in the position is not appended,
        1 and 2 is tag of 'world' and 'python', means the token is appended

    """
    keep_len = True  # keep the length after same to before

    def skip(self, segment, mask_tag, j, **kwargs):
        # do not care what the position token is it if `keep_len=False`
        return self.keep_len and super().skip(segment, mask_tag, j)

    def segment_inplace(self, segment, mask_tag, i, shift, **kwargs):
        j = i + shift
        if self.skip(segment, mask_tag, j):
            return None, shift

        token = self.make_one_token(segment, j)

        if self.keep_len:
            tag = self.word_dict.get(token, self.unk_tag)
            mask_tag[j] = tag
        else:
            j += 1
            self.non_keep_len_inplace(segment, mask_tag, j, token)
            shift += 1

        return j, shift

    def non_keep_len_inplace(self, segment, mask_tag, j, token):
        tag = self.word_dict.get(token, self.unk_tag)
        segment.insert(j, token)
        mask_tag.insert(j, tag)


class RandomAppendTokens(RandomAppend):
    keep_len = False
    n_tokens = 2

    def segment_inplace(self, segment, mask_tag, i, shift, **kwargs):
        assert not self.keep_len, 'Only support `keep_len=False`'

        j = i + shift

        for a in range(self.n_tokens):
            if self.skip(segment, mask_tag, j + a):
                return None, shift

        tokens = self.make_many_tokens(segment, j)

        js = []
        for token in tokens:
            j += 1
            self.non_keep_len_inplace(segment, mask_tag, j, token)
            js.append(j)
            shift += 1

        return js, shift


class RandomDelete(JitterBase):
    """
    ['hello', 'world', 'hello', 'python']
    if keep_len = True:
        -> segment = ['hello', '[DEL]', 'hello', '[DEL]']
        -> mask_tag = [-100, 1, -100, 2]

        -100 is non_mask_tag, means the token in the position is not deleted,
        1 and 2 is tag of 'world' and 'python', means one token is deleted

    if keep_len = False:
        -> segment = ['hello', 'hello']
        -> mask_tag = [1, 2]

        -100 is non_mask_tag, means the token in the position is not deleted,
        1 and 2 is tag of 'world' and 'python', means the token will be got deleted after

    """
    keep_len = True  # keep the length after same to before
    delete_token = '[DEL]'

    def skip(self, segment, mask_tag, j, **kwargs):
        return (
            # keep_len = True
                (self.keep_len and super().skip(segment, mask_tag, j))

                # keep_len = False
                or (not self.keep_len and (
                j + 1 == len(segment)  # if delete the last position token, skip
                or super().skip(segment, mask_tag, j + 1)
        ))
        )

    def segment_inplace(self, segment, mask_tag, i, shift, **kwargs):
        j = i + shift

        if self.skip(segment, mask_tag, j):
            return None, shift

        if self.keep_len:
            token = segment[j]
            tag = self.word_dict.get(token, self.unk_tag)
            segment[j] = self.delete_token
            mask_tag[j] = tag
        else:
            # delete the right site token
            self.non_keep_len_inplace(segment, mask_tag, j)
            shift -= 1

        return j, shift

    def non_keep_len_inplace(self, segment, mask_tag, j):
        token = segment[j + 1]
        tag = self.word_dict.get(token, self.unk_tag)
        mask_tag[j] = tag
        mask_tag.pop(j + 1)
        segment.pop(j + 1)


class RandomDeleteTokens(RandomDelete):
    keep_len = False
    n_tokens = 2

    def segment_inplace(self, segment, mask_tag, i, shift, **kwargs):
        assert not self.keep_len, 'Only support `keep_len=False`'

        j = i + shift

        for a in range(self.n_tokens):
            if self.skip(segment, mask_tag, j + a):
                return None, shift

        js = []
        for a in range(self.n_tokens):
            self.non_keep_len_inplace(segment, mask_tag, j)
            js.append(j)
            j += 1
            shift -= 1

        return js, shift
