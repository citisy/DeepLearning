from collections import defaultdict
from typing import List


def word_id_dict(segments: List[List[str]], start_id=0, filter_func=None):
    """
    Usages:
        >>> segments = [['hello', 'world'], ['hello', 'deep', 'learning']]
        >>> word_id_dict(segments)
        {'hello': 0, 'world': 1, 'deep': 2, 'learning': 3}

        # start with 1, and reserve 0 for unknown word usually
        >>> word_id_dict(segments, start_id=1)
        {'hello': 1, 'world': 2, 'deep': 3, 'learning': 4}
    """
    if filter_func is None:
        filter_func = lambda x: True

    word_dict = dict()
    for line in segments:
        for word in line:
            if word not in word_dict and filter_func(word):
                word_dict[word] = len(word_dict) + start_id

    return word_dict


def word_count_dict(segments: List[List[str]]):
    """
    Usages:
        >>> segments = [['hello', 'world'], ['hello', 'deep', 'learning']]
        >>> word_count_dict(segments)
        {'hello': 2, 'world': 1, 'deep': 1, 'learning': 1}

    """
    word_dict = defaultdict(int)
    for line in segments:
        for word in line:
            word_dict[word] += 1
    return dict(word_dict)
