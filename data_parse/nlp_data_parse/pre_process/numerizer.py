import numpy as np


class KeyValueEncode:
    def __init__(self, word_dict, unk_token='[UNK]'):
        self.word_dict = word_dict
        self.word_inv_dict = {v: k for k, v in word_dict.items()}
        self.unk_token = unk_token
        self.unk_tag = self.word_dict[unk_token]

    def encode(self, segments):
        """
        Usages:
            >>> segments = [['hello', 'world'], ['hello', 'deep', 'learning']]
            >>> word_dict = {'hello': 0, 'world': 1, 'deep': 2, 'learning': 3}
            >>> KeyValueEncode.encode(segments)
            [[0, 1], [0, 2, 3]]
        """
        return [[self.word_dict.get(c, self.unk_tag) for c in seg] for seg in segments]

    def decode(self, tags):
        return [[self.word_inv_dict.get(t, self.unk_token) for t in tag] for tag in tags]


class SeqEncode:
    def __init__(self, sep_token='[SEP]'):
        self.sep_token = sep_token

    def encode(self, segments, new_tag_start=True, start_index=0, max_sep=None):
        """
        Usages:
            >>> segments = [['hello', 'world', '[SEP]', 'hello', 'deep', 'learning'], ['hello', 'world', '[SEP]', 'hello', 'deep', 'learning']]
            >>> SeqEncode().encode(segments)
            [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]

            >>> SeqEncode().encode(segments, new_tag_start=False)
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

                if self.sep_token in s:
                    i = s.index(self.sep_token) + 1
                else:
                    i = len(s)
                tag += [t] * i
                s = s[i:]
                t += 1
                a += 1

            tags.append(tag)
        return tags

    def decode(self, tags):
        pass


class OneHot:
    def __init__(self, word_dict, **kwargs):
        self.word_dict = word_dict
        self.kve = KeyValueEncode(word_dict, **kwargs)

    def encode(self, segments):
        """
        Usages:
            >>> segments = [['hello', 'world'], ['hello', 'deep', 'learning']]
            >>> word_dict = {'hello': 0, 'world': 1, 'deep': 2, 'learning': 3}
            >>> OneHot(word_dict).encode(segments)
            [array([[1, 0, 0, 0],
               [0, 1, 0, 0]]),
            array([[1, 0, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])]
        """
        tmp = self.kve.encode(segments)
        tag = []
        for seg, t in zip(segments, tmp):
            a = np.eye(len(self.word_dict), dtype=int)
            tag.append(a[t])
        return tag

    def decode(self, tags):
        pass


class BytePairEncode:
    """byte pair encode(bpe)
    refer to: https://www.drdobbs.com/a-new-algorithm-for-data-compression/184402829
    """

    def __init__(self, byte_pairs, word_dict, byte_encode_dict=None):
        self.byte_pairs = byte_pairs
        if not isinstance(self.byte_pairs, dict):
            # id: (char1, char2)
            self.byte_pairs = {tuple(c.split()): i for i, c in enumerate(byte_pairs)}

        self.word_dict = word_dict
        self.word_inv_dict = {v: k for k, v in word_dict.items()}

        self.byte_encode_dict = byte_encode_dict or self.make_default_byte_encode_dict()
        self.byte_decoder_dict = {v: k for k, v in self.byte_encode_dict.items()}

        self.caches = {}

    @staticmethod
    def make_default_byte_encode_dict():
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

    def encode(self, segments):
        """byte pair encode
        refer to: https://www.drdobbs.com/a-new-algorithm-for-data-compression/184402829
        """
        tags = []
        for s in segments:
            tag = []

            for word in s:
                # if word in self.caches:
                #     tag = self.caches[word]
                #     continue

                # normalize the segment, fall in [0, 255]
                chars = [self.byte_encode_dict[c] for c in bytearray(word.encode('utf-8'))]

                while len(chars) > 0:
                    min_pair, min_rank = None, float('inf')
                    for i in range(1, len(chars)):
                        pair = (chars[i - 1], chars[i])
                        rank = self.byte_pairs.get(pair, float('inf'))
                        if rank < min_rank:
                            min_rank = rank
                            min_pair = pair
                    if min_pair is None or min_pair not in self.byte_pairs:
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

                tag += [self.word_dict[c] for c in chars]
                self.caches[word] = tag
            tags.append(tag)

        return tags

    def decode(self, tags):
        segments = []
        for tag in tags:
            text = ''.join([self.word_inv_dict[t] for t in tag])
            s = bytearray([self.byte_decoder_dict[byte] for byte in text]).decode('utf-8', errors='replace')
            segments.append(s)
        return segments


class TargetEncode:
    def encode(self, segments):
        pass

    def decode(self, tags):
        pass


class LeaveOneOut:
    def encode(self, segments):
        pass

    def decode(self, tags):
        pass


class WoE:
    def encode(self, segments):
        pass

    def decode(self, tags):
        pass
