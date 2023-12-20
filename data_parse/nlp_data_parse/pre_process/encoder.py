import numpy as np


def simple(segments, word_dict, unk_tag=0):
    """
    Usages:
        >>> segments = [['hello', 'world'], ['hello', 'deep', 'learning']]
        >>> word_dict = {'hello': 0, 'world': 1, 'deep': 2, 'learning': 3}
        >>> simple(segments, word_dict)
        [[0, 1], [0, 2, 3]]
    """
    return [[word_dict.get(c, unk_tag) for c in seg] for seg in segments]


def seq_encode(segments, sep_token, new_tag_start=True, start_index=0, max_sep=None):
    """
    Usages:
        >>> segments = [['hello', 'world', '[SEP]', 'hello', 'deep', 'learning'], ['hello', 'world', '[SEP]', 'hello', 'deep', 'learning']]
        >>> seq_encode(segments, sep_token='[SEP]')
        [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]

        >>> seq_encode(segments, sep_token='[SEP]', new_tag_start=False)
        [[0, 0, 0, 1, 1, 1], [2, 2, 2, 3, 3, 3]]
    """
    max_sep = max_sep or float('inf')
    tags = []
    t = start_index
    for s in segments:
        tag = []
        if new_tag_start:
            t = start_index
        a = 0
        while s:
            if a >= max_sep:
                i = len(s)
                tag += [t - 1] * i
                break

            if sep_token in s:
                i = s.index(sep_token) + 1
            else:
                i = len(s)
            tag += [t] * i
            s = s[i:]
            t += 1
            a += 1

        tags.append(tag)
    return tags


def one_hot(segments, word_dict):
    """
    Usages:
        >>> segments = [['hello', 'world'], ['hello', 'deep', 'learning']]
        >>> word_dict = {'hello': 0, 'world': 1, 'deep': 2, 'learning': 3}
        >>> one_hot(segments, word_dict)
        [array([[1, 0, 0, 0],
           [0, 1, 0, 0]]),
        array([[1, 0, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]])]
    """
    tmp = simple(segments, word_dict)
    tag = []
    for seg, t in zip(segments, tmp):
        a = np.eye(len(word_dict), dtype=int)
        tag.append(a[t])
    return tag


def target_encode(segments, word_dict):
    pass


def leave_one_out(segments, word_dict):
    pass


def woe(segments, word_dict):
    pass
