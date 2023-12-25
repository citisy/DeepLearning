"""context split to different fine-grained type

article (str): all lines fatten to a string
segments (List[List[str]]): all lines after cut
paragraphs (List[str]): all original lines, the lines contain the sep symbol
chunks (List[str]): each line has the same length possibly
"""
import re
import unicodedata
from typing import List
from utils.excluded.charset_dict import utf8_int_dict


def paragraphs_from_article(article: str, sep='\n') -> List[str]:
    return [line + sep for line in article.split(sep)]


def article_from_paragraphs(paragraphs: List[str]) -> str:
    return ''.join(paragraphs)


def filter_text(s, filter_blank=False, filter_pattern=None, keep_pattern=None):
    if filter_blank:
        s = ''.join(s.split())

    if filter_pattern:
        if isinstance(filter_pattern, (list, tuple)):
            filter_pattern = re.compile('|'.join([i.pattern for i in filter_pattern]))
        s = re.sub(filter_pattern, '', s)

    if keep_pattern:
        if isinstance(keep_pattern, (list, tuple)):
            keep_pattern = re.compile('|'.join([i.pattern for i in keep_pattern]))
        s = ''.join(re.findall(keep_pattern, s))

    return s


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


def split_punctuation(seg):
    output = []
    for text in seg:
        chars = list(text)
        i = 0
        start_new_word = True
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

    return [''.join(x) for x in output]


def word_piece(seg, vocab):
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
                if substr in vocab:
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


def segments_from_paragraphs(
        paragraphs: List[str], sep=None, is_split_punctuation=True,
        is_word_piece=False, vocab=None, **filter_kwargs
) -> List[List[str]]:
    """
    Args:
        paragraphs:
        sep:
        is_split_punctuation:
        is_word_piece:
        vocab: for `word_piece`
        **filter_kwargs: kwargs for `filter_text()`

    Usage:
        >>> segments_from_paragraphs(['hello world!'])
        [['hello', 'world', '!']]

        >>> segments_from_paragraphs(['hello world!'], filter_blank=True)
        [['helloworld', '!']]

        >>> from utils.excluded.charset_dict import utf8_pattern_dict
        >>> segments_from_paragraphs(['hello world!'], filter_pattern=utf8_pattern_dict['en_pr'], is_split_punctuation=False)
        [['hello', 'world']]

        >>> segments_from_paragraphs(['hello world!'], keep_pattern=(utf8_pattern_dict['en'], re.compile(' ')))
        [['hello', 'world']]

    """
    segments = []
    for line in paragraphs:
        line = filter_text(line, **filter_kwargs)
        if sep == '':
            seg = list(line)
        else:
            seg = line.split(sep)

        if is_split_punctuation:
            seg = split_punctuation(seg)

        if is_word_piece:
            seg = word_piece(seg, vocab)

        segments.append(seg)

    return segments


def segments_from_paragraphs_by_jieba(paragraphs: List[str], **filter_kwargs) -> List[List[str]]:
    """
    Usage:
        >>> segments_from_paragraphs_by_jieba(['你好 世界！'])
        [['你好', ' ', '世界', '！']]

        >>> segments_from_paragraphs_by_jieba(['你好 世界！'], filter_blank=True)
        [['你好', '世界', '！']]

        >>> from utils.excluded.charset_dict import utf8_pattern_dict
        >>> segments_from_paragraphs_by_jieba(['你好 世界！'], filter_pattern=(utf8_pattern_dict['cjk_pr'], utf8_pattern_dict['en_pr_double']))
        [['你好', ' ', '世界']]

        >>> segments_from_paragraphs_by_jieba(['你好 世界！'], keep_pattern=utf8_pattern_dict['zh'])
        [['你好', '世界']]
    """
    import jieba

    paragraphs = map(lambda x: filter_text(x, **filter_kwargs), paragraphs)
    segments = map(jieba.lcut, paragraphs)

    return list(segments)


def paragraphs_from_segments(segments: List[List[str]], sep='') -> List[str]:
    return [sep.join(p) for p in segments]


def chunks_from_paragraphs(paragraphs: List[str], max_length=512):
    """
    Usage:
        >>> chunks_from_paragraphs(['abcdefghijklmn'], max_length=5)
        ['abcde', 'fghij', 'klmn']

        >>> chunks_from_paragraphs(['abc,def.ghijk,lmn.'], max_length=5)
        ['abc,', 'def.', 'ghijk', ',lmn.']
    """
    chunks = []
    chunk = ''

    for p in paragraphs:
        chunk += p

        if len(chunk) > max_length:
            _chunks = chunks_from_line(chunk, max_length=max_length)
            chunks.extend(_chunks[:-1])
            chunk = _chunks[-1]

    if chunk:
        chunks.append(chunk)

    return chunks


full_stop_rx = re.compile(r'.*[。\.!?！？]', re.DOTALL)
half_stop_rx = re.compile(r'.*[\];；,，、》）}]', re.DOTALL)
newline_stop_rx = re.compile(r'.+\n', re.DOTALL)


def chunks_from_line(line: str, max_length=512) -> List[str]:
    chunks = []
    rest = line

    while True:
        if len(rest) <= max_length:
            if rest:
                chunks.append(rest)
            break

        tail = rest[max_length:]

        left_f, right_f, is_matched_f = truncate_by_stop_symbol(rest[:max_length], full_stop_rx)
        left_n, right_n, is_matched_n = truncate_by_stop_symbol(rest[:max_length], newline_stop_rx)

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
            left_h, right_h, is_matched_h = truncate_by_stop_symbol(rest[:max_length], half_stop_rx)
            if is_matched_h:
                sect, rest = left_h, right_h + tail
            else:
                sect, rest = rest[:max_length], rest[max_length:]

        chunks.append(sect)

    return chunks


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
