import copy
import unicodedata
import numpy as np
from . import BertVocabOp


def lower(paragraphs):
    return [i.lower() for i in paragraphs]


def strip_accents(paragraphs):
    """Strips accents from a piece of text.
    e.g.: 'ü' -> 'u'"""
    r = []
    for text in paragraphs:
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        r.append("".join(output))
    return r


def add_token(segments, start_token=None, end_token=None):
    if start_token is not None:
        segments = [[start_token] + s for s in segments]

    if end_token is not None:
        segments = [s + [end_token] for s in segments]

    return segments


def random_mask(segments, word_dict, unk_tag, mask_token=BertVocabOp.sp_token_dict['mask'], non_mask_tag=-100, mask_prob=0.15):
    segments = copy.deepcopy(segments)
    vocab = list(word_dict.keys())
    mask_tags = []

    for segment in segments:
        mask_probs = np.random.uniform(0., 1., len(segment))
        mask_tag = np.where(mask_probs < mask_prob, -1, non_mask_tag)

        for i, word in enumerate(segment):
            if mask_tag[i] == -1:  # 15% to mask
                prob = mask_probs[i] / mask_prob

                # 80% to add [MASK] token
                if prob < 0.8:
                    segment[i] = mask_token

                # 10% to change to another word
                elif prob < 0.9:
                    segment[i] = np.random.choice(vocab)

                mask_tag[i] = word_dict.get(word, unk_tag)
        mask_tag = mask_tag.tolist()
        mask_tags.append(mask_tag)

    return segments, mask_tags


def joint(text_pairs, sep_token=BertVocabOp.sp_token_dict['sep'], keep_end=True):
    segments = []
    for segments_pair in text_pairs:
        seg = []
        for s in segments_pair:
            seg += s + [sep_token]
        if not keep_end:
            seg = seg[:-1]
        segments.append(seg)
    return segments


def truncate(segments, seq_len):
    return [s[:seq_len] for s in segments]


def pad(segments, max_seq_len, pad_token=BertVocabOp.sp_token_dict['pad']):
    _segments = []
    for s in segments:
        pad_len = max_seq_len - len(s)
        _segments.append(s + [pad_token] * pad_len)
    return _segments


def align(segments, max_seq_len, start_token=None, end_token=None, auto_pad=True, pad_token=BertVocabOp.sp_token_dict['pad']):
    """[[seg]] -> [[start_token], [seg], [end_token], [pad_token]]"""
    segments = add_token(segments, start_token=start_token, end_token=end_token)
    segments = truncate(segments, seq_len=max_seq_len)
    if auto_pad:
        max_seq_len = min(max(len(s) for s in segments), max_seq_len)
    segments = pad(segments, max_seq_len=max_seq_len, pad_token=pad_token)
    return segments
