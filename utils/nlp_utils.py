from typing import List
import re


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
    """
    Examples
        >>> words = [['a', 'b', 'c'], ['a', '{}', 'd']]
        >>> tree = PrefixTree(words, [0, 1])
        >>> tree.get('abc')
        0
        >>> tree.get('abcd')
        1
        >>> tree.get('abcde')
        None

    Notes
        the last token can not be wildcard!
    """

    def __init__(self, words, values=None, unique_value=True,
                 wildcard_pattern=r'\{.*?\}', wildcard_token='[WILD]', match_token='[MATCH]'):
        self.unique_value = unique_value
        self.wildcard_pattern = re.compile(wildcard_pattern)
        self.wildcard_token = wildcard_token
        self.match_token = match_token
        self.tree = dict()

        if isinstance(words, dict):
            words, values = words.keys(), words.values()

        self.build(words, values)

    def build(self, words, values=None):
        for i, word in enumerate(words):
            value = values[i] if values else None
            self.update(word, value)

    def get(self, word, default=None, return_trace=False, return_last=False):
        if not word:
            return default

        tmp = self.tree
        flag = True  # match char flag, not wildcard
        i, last, last_wilds = 0, default, []
        best_i, best_last = i, last
        while i < len(word):
            w = word[i]
            next_tmp = tmp.get(w)

            last_wild = tmp.get(self.wildcard_token)
            if last_wild:
                last_wilds.append([i, last_wild])

            if next_tmp is None:  # search fail
                if len(last_wilds):  # search wildcard
                    i, v = last_wilds[-1]
                    w = word[i]
                    n = v.get(w)
                    if n:
                        next_tmp = n
                        # last_wilds.pop(-1)
                    else:  # wildcard match
                        next_tmp = v
                        flag = False
                        last = None

                else:  # match fail, return
                    tmp = None
                    break

                if last_wilds:
                    last_wilds[-1][0] = i + 1

            last = next_tmp.get(self.match_token, last)
            tmp = next_tmp
            i += 1

            if flag and i > best_i and last:  # match char possible
                best_i = i
                best_last = last

            if i == len(word) and last_wilds:
                last_wilds.pop()
                if last_wilds:
                    last_wilds[-1][0] += 1
                    i, tmp = last_wilds[-1]

            flag = True

        if return_trace:
            r = word[:best_i]
        elif return_last:
            r = best_last
        else:
            r = default

        if tmp is None:
            return r
        else:
            return tmp.get(self.match_token, r)

    def update(self, word, value=None):
        this_dict = self.tree

        for cid, char in enumerate(word):
            if self.wildcard_pattern.match(char):
                char = self.wildcard_token

            this_dict = this_dict.setdefault(char, dict())

            if cid == len(word) - 1:  # last one
                if value is not None:
                    if self.unique_value:
                        this_dict[self.match_token] = value
                    else:
                        this_dict.setdefault(self.match_token, []).append(value)
                else:
                    this_dict[self.match_token] = True
