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


def make_byte_encode_dict():
    # Meaningful token's id
    ids = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))

    # id: char
    byte_encode_dict = {i: chr(i) for i in ids}

    # fill the empty ids with token in [256, inf]
    n = 0
    max_id = 2 ** 8
    for i in range(max_id):
        if i not in byte_encode_dict:
            byte_encode_dict[i] = chr(2 ** 8 + n)
            n += 1
    return byte_encode_dict


def bpe(segments, byte_pairs, word_dict, byte_encode_dict=None):
    """byte pair encode
    refer to: https://www.drdobbs.com/a-new-algorithm-for-data-compression/184402829
    """
    if byte_encode_dict is None:
        byte_encode_dict = make_byte_encode_dict()

    if not isinstance(byte_pairs, dict):
        # id: (char1, char2)
        byte_pairs = {k: i for i, k in enumerate(byte_pairs)}

    tags = []
    for s in segments:
        tag = []

        for word in s:
            # normalize the segment, fall in [0, 255]
            chars = [byte_encode_dict[c] for c in bytearray(word.encode('utf-8'))]

            while len(chars) > 0:
                min_pair, min_rank = None, float('inf')
                for i in range(1, len(chars)):
                    pair = (chars[i - 1], chars[i])
                    rank = byte_pairs.get(pair, float('inf'))
                    if rank < min_rank:
                        min_rank = rank
                        min_pair = pair
                if min_pair is None or min_pair not in byte_pairs:
                    break
                last, tail = chars[0], 1
                for index in range(1, len(chars)):
                    if (last, chars[index]) == min_pair:
                        chars[tail - 1] = last + chars[index]
                        last = last + chars[index]
                    else:
                        chars[tail - 1] = last
                        tail += 1
                        last = chars[index]
                chars[tail - 1] = last
                chars = chars[:tail]

            tag += [word_dict[c] for c in chars]
        tags.append(tag)

    return tags


def target_encode(segments, word_dict):
    pass


def leave_one_out(segments, word_dict):
    pass


def woe(segments, word_dict):
    pass
