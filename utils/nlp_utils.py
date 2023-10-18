from typing import List
import re
from collections import defaultdict
from .excluded import convert_dict
import numpy as np
import torch


class Cutter:
    @staticmethod
    def cut_word_to_char(segments, filter_blank=True):
        _segments = []
        for line in segments:
            if filter_blank:
                line = line.split()
            else:
                line = list(line)
            _segments.append(line)

        return _segments

    @staticmethod
    def cut_word_by_jieba(segments, filter_blank=True) -> List[List[str]]:
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


class Sequencer:
    """make the single words to the sentence sequence
    refer to [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)"""

    @staticmethod
    def n_grams(segments, n_gram=2):
        """n grams

        Args:
            segments (List[List[str]]): should be cut
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


class FineGrainedSpliter:
    """article type split to different fine-grained type

    article (str): all lines fatten to a string
    paragraphs (List[List[str]]):
    chunks (List[List[str]]): each line has the same length possibly
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
    """label text with ids, use key pairs of (acid, ), (pid, pcid) or (cid, ccid) to position a char,
    easy to search chars from different fine-grained text type

    article (str): all lines fatten to a string
    paragraphs (List[List[str]]):
    chunks (List[List[str]]): each line has the same length possibly

    acid: article char id
    pid: paragraph line id
    pcid: paragraph char id
    cid: chunk line id
    ccid: chunk char id
    """

    @staticmethod
    def get_full_id(
            acid=None, pid_pcid=None, cid_ccid=None,
            acid2pid_pcid=None, acid2cid_ccid=None, pid_pcid2acid=None, cid_ccid2acid=None,
    ):
        """input one id pairs of article, paragraphs or chunks, output full id pairs of others"""
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

    @classmethod
    def full_id_pairs(cls, paragraphs, chunks):
        r = dict()
        r.update(cls.article_paragraphs_id_pairs(paragraphs))
        r.update(cls.article_chunks_id_pairs(chunks))
        r.update(cls.paragraphs_chunks_id_pairs(paragraphs, chunks, r['acid2pid_pcid'], r['acid2cid_ccid']))

        return r

    @classmethod
    def article_paragraphs_id_pairs(cls, paragraphs):
        acid2pid_pcid = cls._gen(paragraphs)
        pid_pcid2acid = {v: k for k, v in acid2pid_pcid.items()}

        return dict(
            acid2pid_pcid=acid2pid_pcid,
            pid_pcid2acid=pid_pcid2acid
        )

    @classmethod
    def article_chunks_id_pairs(cls, chunks):
        acid2cid_ccid = cls._gen(chunks)
        cid_ccid2acid = {v: k for k, v in acid2cid_ccid.items()}

        return dict(
            acid2cid_ccid=acid2cid_ccid,
            cid_ccid2acid=cid_ccid2acid
        )

    @classmethod
    def paragraphs_chunks_id_pairs(cls, paragraphs, chunks, acid2pid_pcid=None, acid2cid_ccid=None):
        if acid2pid_pcid is None:
            r = cls.article_paragraphs_id_pairs(paragraphs)
            acid2pid_pcid = r['acid2pid_pcid']

        if acid2cid_ccid is None:
            r = cls.article_chunks_id_pairs(chunks)
            acid2cid_ccid = r['acid2cid_ccid']

        pid_pcid2cid_ccid = {}
        cid_ccid2pid_pcid = {}
        for k in acid2pid_pcid:
            pid_pcid = acid2pid_pcid[k]
            cid_ccid = acid2cid_ccid[k]
            pid_pcid2cid_ccid[pid_pcid] = cid_ccid
            cid_ccid2pid_pcid[cid_ccid] = pid_pcid

        return dict(
            pid_pcid2cid_ccid=pid_pcid2cid_ccid,
            cid_ccid2pid_pcid=cid_ccid2pid_pcid
        )

    @staticmethod
    def _gen(lines):
        id_dic = {}
        base_id = 0
        for line_id, p in enumerate(lines):
            for char_id, _ in enumerate(p):
                id_dic[base_id] = (line_id, char_id)
                base_id += 1

        return id_dic


class DictMaker:
    @staticmethod
    def word_id_dict(segments):
        """

        Args:
            segments (List[List[str]]):

        Returns:
            a dict like {word: id}
        """
        word_dict = dict()
        for line in segments:
            for word in line:
                if word not in word_dict:
                    # start with 1, and reserve 0 for unknown word usually
                    word_dict[word] = len(word_dict) + 1

        return word_dict

    @staticmethod
    def word_count_dict(segments):
        """

        Args:
            segments (List[List[str]]):

        Returns:
            a dict like {word: count}
        """
        word_dict = defaultdict(int)
        for line in segments:
            for word in line:
                word_dict[word] += 1
        return word_dict


class Encoder:
    @staticmethod
    def simple(segments, word_dict):
        pass

    @staticmethod
    def one_hot(segments, word_id_dict):
        pass


class Converter:
    @staticmethod
    def num(s):
        pass

    @staticmethod
    def round_to_half_punctuation(s: str):
        for a, b in convert_dict.round_to_half_punctuation.items():
            s = s.replace(a, b)
        return s

    @staticmethod
    def _convert(s: str, d: dict):
        for a, b in d.items():
            s = s.replace(a, b)
        return s


class Decoder:
    @staticmethod
    def enum_search(x):
        pass

    @staticmethod
    def greedy_search(x):
        """

        Args:
            x (torch.Tensor): (batch_size, seq_length, vocab_size) after log softmax

        Returns:
            preds: (batch_size, beam_size, seq_length)
            probs: (batch_size, beam_size)

        """
        probs, preds = x.max(2)
        return preds, probs

    @staticmethod
    def beam_search(x, beam_size=4):
        """

        Args:
            x (torch.Tensor): (batch_size, seq_length, vocab_size) after log softmax
            beam_size:

        Returns:
            preds: (batch_size, beam_size, seq_length)
            probs: (batch_size, beam_size)

        """
        batch, seq_len, vocab_size = x.shape
        probs, pred = x[:, 0, :].topk(beam_size, sorted=True)
        preds = pred.unsqueeze(-1)
        for i in range(1, seq_len):
            probs = probs.unsqueeze(-1) + x[:, i, :].unsqueeze(1).repeat(1, beam_size, 1)
            probs, pred = probs.view(batch, -1).topk(beam_size, sorted=True)
            idx = torch.div(pred, vocab_size, rounding_mode='trunc')
            pred = pred % vocab_size
            preds = torch.gather(preds, 1, idx.unsqueeze(-1).repeat(1, 1, i))
            preds = torch.cat([preds, pred.unsqueeze(-1)], dim=-1)
        return preds, probs

    @staticmethod
    def prefix_beam_search(x):
        pass
