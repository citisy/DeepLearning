import numpy as np


def cut_word_by_jieba(segments, filter_blank=True):
    """

    Args:
        segments (List[str]):
        filter_blank (bool):

    Returns:

    """
    import jieba

    segments = map(jieba.lcut, segments)

    if filter_blank:
        segments = map(lambda x: ' '.join(x).split(), segments)

    return list(segments)


class Sequence:
    """refer to: https://aclanthology.org/W04-1013.pdf"""

    @staticmethod
    def n_grams(segments, n_gram=2):
        """n grams

        Args:
            segments (List[list]): should be cut
            n_gram (int):

        Returns:
            List[set]
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
    def longest_common_subsequence(cls, a, b):
        """longest common subsequence(LCS)

        Args:
            a:
            b:

        Returns:

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
    def weighted_longest_common_subsequence(cls, a, b, f=lambda x: x ** 2, f_inv=lambda x: x ** 0.5):
        """weighted longest common subsequence(WLCS)
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
