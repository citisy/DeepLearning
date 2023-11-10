import numpy as np


def simple(segments, word_dict):
    """
    Usages:
        >>> segments = [['hello', 'world'], ['hello', 'deep', 'learning']]
        >>> word_dict = {'hello': 0, 'world': 1, 'deep': 2, 'learning': 3}
        >>> simple(segments, word_dict)
        [[0, 1], [0, 2, 3]]
    """
    return [[word_dict.get(c, 0) for c in seg] for seg in segments]


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
    trans = []
    for seg, t in zip(segments, tmp):
        a = np.eye(len(word_dict), dtype=int)
        trans.append(a[t])
    return trans
