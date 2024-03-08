"""context split to different fine-grained type

article (str):
    all lines fatten to a string
    e.g.: 'hello world! hello python!'
paragraphs (List[str]):
    all original lines, each itme in list is a str line
    e.g.: ['hello world!', 'hello python!']
paragraph (str)
    one paragraph can be an article
chunked_paragraphs (List[str]):
    each line has the same length as paragraphs as possibly, each itme in list is a str line
    e.g.: ['hello world! hello python!']
segments (List[List[str]]):
    all lines after cut, each itme in list is a cut word list
    e.g.: [['hello', 'world!'], ['hello', 'python!']]
segment (List[str])
    one segment can be a paragraphs
chunked_segments (List[List[str]]):
    each line has the same length as segments as possibly, each itme in list is a cut word list
    e.g.: [['hello', 'world!', 'hello', 'python!']]
"""
import re
import unicodedata
from typing import List
from tqdm import tqdm
from utils.excluded.charset_dict import utf8_int_dict
from utils.visualize import TextVisualize


class ToArticle:
    def __init__(self, sep='\n'):
        self.sep = sep

    def article_from_paragraphs(self, paragraphs: List[str]) -> str:
        return self.sep.join(paragraphs)


class ToParagraphs:
    def __init__(self, sep='\n', keep_sep=False):
        self.sep = sep
        self.keep_sep = keep_sep

    def from_article(self, article: str) -> List[str]:
        if self.keep_sep:
            paragraphs = [line + self.sep for line in article.split(self.sep)]
        else:
            paragraphs = article.split(self.sep)

        return paragraphs

    def from_segments(self, segments: List[List[str]]) -> List[str]:
        return [self.sep.join(s) for s in segments]


class ToSegment:
    """
    Usage:
        >>> text = 'hello [MASK]!'
        >>> ToSegment().from_paragraph(text)
        ['hello', '[', 'MASK', ']', '!']

        >>> ToSegment(sep='').from_paragraph(text)
        ['h', 'e', 'l', 'l', 'o', ' ', '[', 'M', 'A', 'S', 'K', ']', '!']

        >>> ToSegment(sp_tokens=['[MASK]']).from_paragraph(text)
        ['hello', '[MASK]', '!']
    """
    def __init__(
            self,
            sep=None, sep_pattern=None,
            sp_tokens=(), cleaner=None,
            is_split_punctuation=True, is_word_piece=False, vocab=None,
            **kwargs
    ):
        """

        Args:
            sep: split seq symbol
            sep_pattern (str or re.Pattern): re pattern seq
            is_split_punctuation:
            is_word_piece:
            vocab: for `word_piece`
        """
        sp_pattern = []
        for s in sp_tokens:
            for a in '\\[]{}.*?':
                s = s.replace(a, '\\' + a)
            sp_pattern.append(s)
        self.sp_tokens = sp_tokens
        self.sp_pattern = re.compile('|'.join(sp_pattern))

        self.sep = sep
        self.sep_pattern = sep_pattern
        self.cleaner = cleaner
        self.vocab = vocab

        self.deep_split_funcs = []
        if is_split_punctuation:
            self.deep_split_funcs.append(self.from_paragraph_with_punctuation)
        if is_word_piece:
            self.deep_split_funcs.append(self.from_paragraph_with_word_piece)

    def from_paragraph(self, paragraph):
        if self.sp_tokens:
            _segment = self.from_paragraph_with_sp_tokens(paragraph)
        else:
            _segment = [paragraph]

        segment = []
        for text in _segment:
            if text in self.sp_tokens:
                segment.append(text)
            else:
                if self.cleaner:
                    text = self.cleaner(text)
                seg = self.shallow_split(text)
                seg = self.deep_split(seg)
                segment += seg
        return segment

    def from_paragraph_with_sp_tokens(self, paragraph):
        i = 0
        segment = []
        while i < len(paragraph):
            r = self.sp_pattern.search(paragraph[i:])
            if r:
                span = r.span()
                segment.append(paragraph[i: span[0]])
                segment.append(paragraph[span[0]: span[1]])
                i = span[1]
            else:
                segment.append(paragraph[i:])
                break

        return segment

    def shallow_split(self, paragraph):
        if self.sep == '':
            segment = list(paragraph)
        elif self.sep is not None:
            segment = paragraph.split(self.sep)
        elif self.sep_pattern is not None:
            if hasattr(self.sep_pattern, 'findall'):
                segment = self.sep_pattern.findall(paragraph)
            else:
                segment = re.findall(self.sep_pattern, paragraph)
        else:
            segment = paragraph.split(self.sep)
        return segment

    def deep_split(self, segment):
        def split(segment, func):
            _segment = []
            for seg in segment:
                _segment += func(seg)
            return _segment

        for func in self.deep_split_funcs:
            segment = split(segment, func)

        return segment

    @staticmethod
    def _is_punctuation(char):
        """Checks whether `char` is a punctuation character.
        treat all non-letter/number ASCII as punctuation"""
        cp = ord(char)
        for span in utf8_int_dict['en_pr']:
            if span[0] <= cp <= span[1]:
                return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def from_paragraph_with_punctuation(self, text):
        output = []
        # for text in seg:
        chars = list(text)
        i = 0
        start_new_word = True
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return [''.join(x) for x in output]

    def from_paragraph_with_word_piece(self, paragraph):
        chars = list(paragraph)
        is_bad = False
        start = 0
        segment = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            segment.append(cur_substr)
            start = end

        if is_bad:
            segment = [paragraph]

        return segment


class ToSegments:
    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose
        self.to_segment = ToSegment(**kwargs)

    def from_paragraphs(self, paragraphs: List[str]) -> List[List[str]]:
        """see also cleaner
        Usage:
            >>> ToSegments().from_paragraphs(['hello world!'])
            [['hello', 'world', '!']]
        """
        segments = []
        if self.verbose:
            paragraphs = tqdm(paragraphs, desc=TextVisualize.highlight_str('Split paragraphs'))

        for line in paragraphs:
            seg = self.to_segment.from_paragraph(line)
            segments.append(seg)

        return segments

    def from_paragraphs_by_jieba(self, paragraphs: List[str]) -> List[List[str]]:
        """see also cleaner
        Usage:
            >>> ToSegments().from_paragraphs_by_jieba(['你好 世界！'])
            [['你好', ' ', '世界', '！']]
        """
        import jieba

        paragraphs = map(self.to_segment.cleaner, paragraphs)
        segments = map(jieba.lcut, paragraphs)
        segments = map(self.to_segment.deep_split, segments)

        return list(segments)
