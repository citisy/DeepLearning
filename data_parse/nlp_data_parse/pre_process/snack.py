def add_token(seqs, start_obj=None, end_obj=None):
    if start_obj is not None:
        seqs = [[start_obj] + s for s in seqs]

    if end_obj is not None:
        seqs = [s + [end_obj] for s in seqs]

    return seqs


def joint(seq_pairs, sep_obj=None, keep_end=True):
    segments = []
    for segments_pair in seq_pairs:
        seg = []
        for s in segments_pair:
            seg += s + [sep_obj]
        if not keep_end:
            seg = seg[:-1]
        segments.append(seg)
    return segments


def truncate(seqs, seq_len):
    return [s[:seq_len] for s in seqs]


def pad(seqs, max_seq_len, pad_obj=None):
    if pad_obj is None:
        return seqs

    _segments = []
    for s in seqs:
        pad_len = max_seq_len - len(s)
        _segments.append(s + [pad_obj] * pad_len)
    return _segments


def align(seqs, max_seq_len, start_obj=None, end_obj=None, auto_pad=True, pad_obj=None):
    """all sequence in sequences has the same length,
    e.g.: [[seg]] -> [[start_obj], [seg], [end_obj], [pad_obj]]"""
    seqs = add_token(seqs, start_obj=start_obj, end_obj=end_obj)
    seqs = truncate(seqs, seq_len=max_seq_len)
    if auto_pad:
        max_seq_len = min(max(len(s) for s in seqs), max_seq_len)
    seqs = pad(seqs, max_seq_len=max_seq_len, pad_obj=pad_obj)
    return seqs
