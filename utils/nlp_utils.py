import numpy as np
from typing import List
import re


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
    """[ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)"""

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


class LevelGeneration:
    """
    article: '...', all lines fatten to a string
    paragraphs: [[], [], ....]
    chunks: [[], [], ....], each line has the same length possibly
    """
    def __init__(self, max_length=512, line_end='\n'):
        self.max_length = max_length
        self.line_end = line_end
        self.full_stop_rx = re.compile(r'.*(。|\.{3,6})(?![.。])', re.DOTALL)
        self.half_stop_rx = re.compile(r'.*[];；,，、》）}!?！？]', re.DOTALL)
        self.newline_stop_rx = re.compile(f'.+{line_end}', re.DOTALL)

    def gen_paragraphs(self, lines):
        for i, line in enumerate(lines):
            lines[i] = line + self.line_end

    def gen_article(self, paragraphs):
        return ''.join(paragraphs)

    def gen_chunks(self, paragraphs):
        chunks = []
        chunk = ''

        for p in paragraphs:
            chunk += p

            if len(chunk) > self.max_length:
                segs = self.segment_chunks(chunk)
                chunks.extend(segs[:-1])
                chunk = segs[-1]

        if chunk:
            chunks.append(chunk)

        return chunks

    def segment_chunks(self, text):
        segs = []
        rest = text

        while True:
            if len(rest) <= self.max_length:
                if rest:
                    segs.append(rest)
                break

            sect, rest = self.segment_one_chunk(rest)
            segs.append(sect)

        return segs

    def segment_one_chunk(self, text) -> tuple:
        if len(text) <= self.max_length:
            return text, ''

        tailing = text[self.max_length:]

        left_f, righ_f, is_matched_f = self.truncate_by_stop_symbol(text[:self.max_length], self.full_stop_rx)
        left_n, righ_n, is_matched_n = self.truncate_by_stop_symbol(text[:self.max_length], self.newline_stop_rx)

        if is_matched_f and is_matched_n:
            if len(left_f) >= len(left_n):
                return left_f, righ_f + tailing
            else:
                return left_n, righ_n + tailing
        elif is_matched_f:
            return left_f, righ_f + tailing
        elif is_matched_n:
            return left_n, righ_n + tailing

        left_h, righ_h, is_matched_h = self.truncate_by_stop_symbol(text[:self.max_length], self.half_stop_rx)
        if is_matched_h:
            return left_h, righ_h + tailing

        return text[:self.max_length], text[self.max_length:]

    def truncate_by_stop_symbol(self, text, pattern: re.Pattern) -> tuple:
        m = pattern.match(text)

        if m:
            left = text[:m.span()[1]]
            right = text[m.span()[1]:]
            is_matched = True
        else:
            left, right = text, ''
            is_matched = False

        return left, right, is_matched


class IdMapper:
    """
    article: '...', all lines fatten to a string
    paragraphs: [[], [], ....]
    chunks: [[], [], ....], each line has the same length possibly

    acid: id of article
    pid: id of paragraph line
    pcid: id of paragraph char
    cid: id of chunk line
    ccid: id of chunk char
    """

    def gen_full_ids(self, paragraphs, chunks):
        r = dict()
        r.update(self.gen_article_paragraphs_ids(paragraphs))
        r.update(self.gen_article_chunks_ids(chunks))

        return r

    def gen_article_paragraphs_ids(self, paragraphs):
        acid2pid_pcid = self._gen(paragraphs)
        pid_pcid2acid = {v: k for k, v in acid2pid_pcid.items()}

        return dict(
            acid2pid_pcid=acid2pid_pcid,
            pid_pcid2acid=pid_pcid2acid
        )

    def gen_article_chunks_ids(self, chunks):
        acid2cid_ccid = self._gen(chunks)
        cid_ccid2acid = {v: k for k, v in acid2cid_ccid.items()}

        return dict(
            acid2cid_ccid=acid2cid_ccid,
            cid_ccid2acid=cid_ccid2acid
        )

    def _gen(self, lines):
        id_dic = {}
        base_id = 0
        for line_id, p in enumerate(lines):
            for char_id, _ in enumerate(p):
                id_dic[base_id] = (line_id, char_id)
                base_id += 1

        return id_dic

    def get_full_id(
            self,
            acid=None, pid_pcid=None, cid_ccid=None,
            acid2pid_pcid=None, acid2cid_ccid=None, pid_pcid2acid=None, cid_ccid2acid=None,
    ):
        if acid:
            pid_pcid = acid2pid_pcid[acid]
            cid_ccid = acid2cid_ccid[acid]

        elif pid_pcid:
            acid = pid_pcid2acid[acid]
            cid_ccid = acid2cid_ccid[acid]

        elif cid_ccid:
            acid = cid_ccid2acid[cid_ccid]
            pid_pcid = acid2pid_pcid[acid]

        return dict(
            acid=acid,
            pid_pcid=pid_pcid,
            cid_ccid=cid_ccid
        )
