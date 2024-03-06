"""context split to different fine-grained type

article (str):
    all lines fatten to a string
    e.g.: 'hello world! hello python!'
paragraphs (List[str]):
    all original lines, each itme in list is a str line
    e.g.: ['hello world!', 'hello python!']
chunked_paragraphs (List[str]):
    each line has the same length as paragraphs as possibly, each itme in list is a str line
    e.g.: ['hello world! hello python!']
segments (List[List[str]]):
    all lines after cut, each itme in list is a cut word list
    e.g.: [['hello', 'world!'], ['hello', 'python!']]
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


class ToChunkedParagraphs:
    full_stop_rx = re.compile(r'.*[。\.!?！？]', re.DOTALL)
    half_stop_rx = re.compile(r'.*[\];；,，、》）}]', re.DOTALL)
    newline_stop_rx = re.compile(r'.+\n', re.DOTALL)

    def __init__(self, max_length=512, min_length=None, **kwargs):
        self.max_length = max_length
        self.min_length = min_length
        self.__dict__.update(kwargs)

    def from_paragraphs(self, paragraphs: List[str], ):
        """
        Usage:
            >>> ToChunkedParagraphs(max_length=5).from_paragraphs(['abcdefghijklmn'])
            ['abcde', 'fghij', 'klmn']

            >>> ToChunkedParagraphs(max_length=5).from_paragraphs(['abc', 'def', 'ghi', 'jk', 'lmn'])
            ['abcde', 'fghij', 'klmn']

            >>> ToChunkedParagraphs(max_length=5, min_length=3).from_paragraphs(['abc', 'def', 'ghi', 'jk', 'lmn'])
            ['abc', 'def', 'ghi', 'jklmn']

            >>> ToChunkedParagraphs(max_length=5).from_paragraphs(['abc,def.ghijk,lmn.'])
            ['abc,', 'def.', 'ghijk', ',lmn.']
        """
        chunked_paragraphs = []
        chunk = ''

        for p in paragraphs:
            chunk += p

            if len(chunk) > self.max_length:
                chunks = self.from_line(chunk)
                chunked_paragraphs.extend(chunks[:-1])
                chunk = chunks[-1]
            elif self.min_length and len(chunk) >= self.min_length:
                chunked_paragraphs.append(chunk)
                chunk = ''
                continue

        if chunk:
            chunked_paragraphs.append(chunk)

        return chunked_paragraphs

    def from_line(self, line: str) -> List[str]:
        chunked_paragraphs = []
        rest = line

        while True:
            if len(rest) <= self.max_length:
                if rest:
                    chunked_paragraphs.append(rest)
                break

            tail = rest[self.max_length:]

            left_f, right_f, is_matched_f = self.truncate_by_stop_symbol(rest[:self.max_length], self.full_stop_rx)
            left_n, right_n, is_matched_n = self.truncate_by_stop_symbol(rest[:self.max_length], self.newline_stop_rx)

            if is_matched_f and is_matched_n:
                if len(left_f) >= len(left_n):
                    sect, rest = left_f, right_f + tail
                else:
                    sect, rest = left_n, right_n + tail
            elif is_matched_f:
                sect, rest = left_f, right_f + tail
            elif is_matched_n:
                sect, rest = left_n, right_n + tail
            else:
                left_h, right_h, is_matched_h = self.truncate_by_stop_symbol(rest[:self.max_length], self.half_stop_rx)
                if is_matched_h:
                    sect, rest = left_h, right_h + tail
                else:
                    sect, rest = rest[:self.max_length], rest[self.max_length:]

            chunked_paragraphs.append(sect)

        return chunked_paragraphs

    @staticmethod
    def truncate_by_stop_symbol(line, pattern: re.Pattern) -> tuple:
        m = re.match(pattern, line)

        if m:
            left = line[:m.span()[1]]
            right = line[m.span()[1]:]
            is_matched = True
        else:
            left, right = line, ''
            is_matched = False

        return left, right, is_matched


class ToSegment:
    def __init__(
            self,
            sp_tokens=(),
            filter_blank=False, filter_pattern=None, keep_pattern=None,
            sep=None, sep_pattern=None,
            is_split_punctuation=True, is_word_piece=False, vocab=None,
            **kwargs
    ):
        """

        Args:
            sep: split seq symbol
            sep_pattern (str or re.Pattern): re pattern seq
            is_split_punctuation:
            is_strip_accents:
            is_word_piece:
            vocab: for `word_piece`
            **filter_kwargs: kwargs for `filter_text()`
        """
        sp_pattern = []
        for s in sp_tokens:
            for a in '\\[]{}.*?':
                s = s.replace(a, '\\' + a)
            sp_pattern.append(s)
        self.sp_tokens = sp_tokens
        self.sp_pattern = re.compile('|'.join(sp_pattern))

        self.filter_blank = filter_blank

        if filter_pattern:
            if isinstance(filter_pattern, (list, tuple)):
                filter_pattern = re.compile('|'.join([i.pattern for i in filter_pattern]))
        self.filter_pattern = filter_pattern

        if keep_pattern:
            if isinstance(keep_pattern, (list, tuple)):
                keep_pattern = re.compile('|'.join([i.pattern for i in keep_pattern]))
        self.keep_pattern = keep_pattern

        self.sep = sep
        self.sep_pattern = sep_pattern
        self.is_split_punctuation = is_split_punctuation
        self.is_word_piece = is_word_piece
        self.vocab = vocab

    def from_paragraph(self, paragraph):
        if self.sp_tokens:
            _segment = self.split_sp_tokens(paragraph)
        else:
            _segment = [paragraph]

        segment = []
        for text in _segment:
            if text in self.sp_tokens:
                segment.append(text)
            else:
                text = self.filter_text(text)
                seg = self.split_text(text)
                seg = self.tidy_segment(seg)
                segment += seg
        return segment

    def split_sp_tokens(self, paragraph):
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

    def filter_text(self, text):
        if self.filter_blank:
            text = ''.join(text.split())

        if self.filter_pattern:
            text = re.sub(self.filter_pattern, '', text)

        if self.keep_pattern:
            text = ''.join(re.findall(self.keep_pattern, text))

        return text

    def split_text(self, text):
        if self.sep == '':
            segment = list(text)
        elif self.sep is not None:
            segment = text.split(self.sep)
        elif self.sep_pattern is not None:
            if hasattr(self.sep_pattern, 'findall'):
                segment = self.sep_pattern.findall(text)
            else:
                segment = re.findall(self.sep_pattern, text)
        else:
            segment = text.split(self.sep)
        return segment

    def tidy_segment(self, segment):
        if self.is_split_punctuation:
            segment = self.split_punctuation(segment)

        if self.is_word_piece:
            segment = self.word_piece(segment)

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

    @classmethod
    def split_punctuation(cls, seg):
        output = []
        for text in seg:
            chars = list(text)
            i = 0
            start_new_word = True
            while i < len(chars):
                char = chars[i]
                if cls._is_punctuation(char):
                    output.append([char])
                    start_new_word = True
                else:
                    if start_new_word:
                        output.append([])
                    start_new_word = False
                    output[-1].append(char)
                i += 1

        return [''.join(x) for x in output]

    def word_piece(self, seg):
        output_tokens = []
        for token in seg:
            chars = list(token)
            is_bad = False
            start = 0
            sub_tokens = []
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
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


class ToSegments:
    def __init__(self, verbose=False, **kwargs):
        self.verbose = verbose
        self.to_segment = ToSegment(**kwargs)

    def from_paragraphs(self, paragraphs: List[str]) -> List[List[str]]:
        """
        Usage:
            >>> ToSegments().from_paragraphs(['hello world!'])
            [['hello', 'world', '!']]

            >>> ToSegments(filter_blank=True).from_paragraphs(['hello world!'])
            [['helloworld', '!']]

            >>> from utils.excluded.charset_dict import utf8_pattern_dict
            >>> ToSegments(is_split_punctuation=False).from_paragraphs(['hello world!'], filter_pattern=utf8_pattern_dict['en_pr'])
            [['hello', 'world']]

            >>> ToSegments(keep_pattern=(utf8_pattern_dict['en'], re.compile(' '))).from_paragraphs(['hello world!'])
            [['hello', 'world']]

        """
        segments = []
        if self.verbose:
            paragraphs = tqdm(paragraphs, desc=TextVisualize.highlight_str('Split paragraphs'))

        for line in paragraphs:
            seg = self.to_segment.from_paragraph(line)
            segments.append(seg)

        return segments

    def from_paragraphs_by_jieba(self, paragraphs: List[str]) -> List[List[str]]:
        """
        Usage:
            >>> ToSegments().from_paragraphs_by_jieba(['你好 世界！'])
            [['你好', ' ', '世界', '！']]

            >>> ToSegments(filter_blank=True).from_paragraphs_by_jieba(['你好 世界！'])
            [['你好', '世界', '！']]

            >>> from utils.excluded.charset_dict import utf8_pattern_dict
            >>> ToSegments(filter_pattern=(utf8_pattern_dict['cjk_pr'], utf8_pattern_dict['en_pr_double'])).from_paragraphs_by_jieba(['你好 世界！'])
            [['你好', ' ', '世界']]

            >>> ToSegments(keep_pattern=utf8_pattern_dict['zh']).from_paragraphs_by_jieba(['你好 世界！'])
            [['你好', '世界']]
        """
        import jieba

        paragraphs = map(self.to_segment.filter_text, paragraphs)
        segments = map(jieba.lcut, paragraphs)
        segments = map(self.to_segment.tidy_segment, segments)

        return list(segments)


class ToChunkedSegments:
    full_stop_tokens = set('。.!?！？')
    half_stop_tokens = set('];；,，、》）}')
    newline_stop_tokens = set('\n')

    def __init__(self, max_length=512, min_length=None, verbose=False, **kwargs):
        self.max_length = max_length
        self.min_length = min_length
        self.verbose = verbose

    def from_segments(self, segments: List[List[str]]):
        chunked_segments = []
        chunked_segment = []

        if self.verbose:
            segments = tqdm(segments, desc=TextVisualize.highlight_str('Chunk segments'))

        for p in segments:
            chunked_segment += p

            if len(chunked_segment) > self.max_length:
                chunks = self.from_segment(chunked_segment)
                chunked_segments.extend(chunks[:-1])
                chunked_segment = chunks[-1]
            elif self.min_length and len(chunked_segment) >= self.min_length:
                chunked_segments.append(chunked_segment)
                chunked_segment = []
                continue

        if chunked_segment:
            chunked_segments.append(chunked_segment)

        return chunked_segments

    def from_segment(self, segment):
        chunked_segments = []
        rest = segment

        while True:
            if len(rest) <= self.max_length:
                if rest:
                    chunked_segments.append(rest)
                break

            keep = rest[:self.max_length]
            tail = rest[self.max_length:]

            left_f, right_f, is_matched_f = self.truncate_by_stop_token(keep, self.full_stop_tokens)
            left_n, right_n, is_matched_n = self.truncate_by_stop_token(keep, self.newline_stop_tokens)

            if is_matched_f and is_matched_n:
                if len(left_f) >= len(left_n):
                    sect, rest = left_f, right_f + tail
                else:
                    sect, rest = left_n, right_n + tail
            elif is_matched_f:
                sect, rest = left_f, right_f + tail
            elif is_matched_n:
                sect, rest = left_n, right_n + tail
            else:
                left_h, right_h, is_matched_h = self.truncate_by_stop_token(keep, self.half_stop_tokens)
                if is_matched_h:
                    sect, rest = left_h, right_h + tail
                else:
                    sect, rest = keep, tail

            chunked_segments.append(sect)

        return chunked_segments

    @staticmethod
    def truncate_by_stop_token(segment, token: set) -> tuple:
        is_matched = False
        left, right = segment, []

        for i, s in enumerate(segment):
            if s in token:
                left = segment[:i + 1]
                right = segment[i + 1:]
                is_matched = True

        return left, right, is_matched
