from typing import List


class Sequencer:
    """make the single words to the sentence sequence
    refer to [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)"""

    @staticmethod
    def n_grams(segments: List[List[str]], n_gram=2) -> List[set]:
        """
        Examples:
            >>> Sequencer.n_grams([['a', 'b', 'c', 'd', 'e']])
            [{('d', 'e'), ('a', 'b'), ('b', 'c'), ('c', 'd')}]

            >>> Sequencer.n_grams([['a', 'b', 'c', 'd', 'e']], n_gram=3)
            [{('c', 'd', 'e'), ('b', 'c', 'd'), ('a', 'b', 'c')}]
        """
        segments_n_grams = []
        for seg in segments:
            seg = [tuple(seg[i: i + n_gram]) for i in range(len(seg) - n_gram + 1)]
            seg = set(seg)
            segments_n_grams.append(seg)

        return segments_n_grams

    @staticmethod
    def search_seq(a, b, table):
        def cur(i, j):
            if i == 0 or j == 0:
                return []
            elif a[i - 1] == b[j - 1]:
                return cur(i - 1, j - 1) + [(a[i - 1], i)]
            elif table[i - 1, j] > table[i, j - 1]:
                return cur(i - 1, j)
            else:
                return cur(i, j - 1)

        return list(map(lambda x: x[0], cur(len(a), len(b))))

    @classmethod
    def longest_common_subsequence(cls, a: List[str], b: List[str]) -> dict:
        """longest common subsequence(LCS)

        Examples:
            >>> Sequencer.longest_common_subsequence(['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'b', 'c', 'd', 'e'])
            {'lcs': ['a', 'b', 'c', 'd', 'e'], 'score': 5}
        """
        m, n = len(a), len(b)
        table = dict()
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    table[i, j] = 0
                elif a[i - 1] == b[j - 1]:
                    table[i, j] = table[i - 1, j - 1] + 1
                else:
                    table[i, j] = max(table[i - 1, j], table[i, j - 1])

        return dict(
            lcs=cls.search_seq(a, b, table),
            score=table[m, n]
        )

    @classmethod
    def weighted_longest_common_subsequence(cls, a, b, f=lambda x: x ** 2, f_inv=lambda x: x ** 0.5) -> dict:
        """weighted longest common subsequence(WLCS)

        Examples:
            >>> Sequencer.weighted_longest_common_subsequence(['a', 'b', 'c', 'd', 'e'], ['a', 'b', 'b', 'c', 'd', 'e'])
            {'lcs': ['a', 'b', 'c', 'd', 'e'], 'score': 4.123105625617661}
        """
        m, n = len(a), len(b)
        c, w = dict(), dict()

        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    c[i, j] = 0
                    w[i, j] = 0
                elif a[i - 1] == b[j - 1]:
                    k = w[i - 1, j - 1]
                    c[i, j] = c[i - 1, j - 1] + f(k + 1) - f(k)
                    w[i, j] = k + 1
                elif c[i - 1, j] > c[i, j - 1]:
                    c[i, j] = c[i - 1, j]
                    w[i, j] = 0
                else:
                    c[i, j] = c[i, j - 1]
                    w[i, j] = 0

        return dict(
            lcs=cls.search_seq(a, b, c),
            score=f_inv(c[m, n])
        )


class PrefixTree:
    def __init__(self, words, values=None, unique=False, end_flag=True):
        self.unique = unique
        self.end_flag = end_flag
        self.tree = dict()
        self.build(words, values)

    def build(self, words, values=None):
        for i, word in enumerate(words):
            value = values[i] if values else None
            self.update(word, value)

    def get(self, word, default=None, return_trace=False, return_last=False):
        tmp = self.tree
        last = default
        for i, w in enumerate(word):
            tmp = tmp.get(w)
            if tmp is None:
                if return_trace:
                    return word[:i]
                elif return_last:
                    return last
                else:
                    return default
            else:
                last = tmp.get(self.end_flag, last)

        if return_trace:
            r = word
        elif return_last:
            r = last
        else:
            r = default

        return tmp.get(self.end_flag, r)

    def update(self, word, value=None):
        this_dict = self.tree

        for cid, char in enumerate(word):
            this_dict = this_dict.setdefault(char, dict())

            if cid == len(word) - 1:  # last one
                if value:
                    if self.unique:
                        this_dict[self.end_flag] = value
                    else:
                        this_dict.setdefault(self.end_flag, []).append(value)
                else:
                    this_dict[self.end_flag] = None
