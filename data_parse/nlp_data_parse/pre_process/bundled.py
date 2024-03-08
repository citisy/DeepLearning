import copy
import numpy as np
from . import BertVocabOp


def add_token(segments, start_token=None, end_token=None):
    if start_token is not None:
        segments = [[start_token] + s for s in segments]

    if end_token is not None:
        segments = [s + [end_token] for s in segments]

    return segments


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
    """all segment in segments has the same length,
    [[seg]] -> [[start_token], [seg], [end_token], [pad_token]]"""
    segments = add_token(segments, start_token=start_token, end_token=end_token)
    segments = truncate(segments, seq_len=max_seq_len)
    if auto_pad:
        max_seq_len = min(max(len(s) for s in segments), max_seq_len)
    segments = pad(segments, max_seq_len=max_seq_len, pad_token=pad_token)
    return segments
