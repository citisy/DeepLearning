"""context split to different fine-grained type

article (str): all lines fatten to a string
segments (List[List[str]]): all lines after cut
paragraphs (List[str]): all original lines, the lines contain the sep symbol
chunks (List[str]): each line has the same length possibly
"""
import re
from typing import List


def paragraphs_from_article(article: str, sep='\n') -> List[str]:
    return [line + sep for line in article.split(sep)]


def article_from_paragraphs(paragraphs: List[str]) -> str:
    return ''.join(paragraphs)


def filter_func(s, filter_blank=False, filter_pattern=None, keep_pattern=None):
    if filter_blank:
        s = ''.join(s.split())

    if filter_pattern:
        if isinstance(filter_pattern, (list, tuple)):
            for p in filter_pattern:
                s = re.sub(p, '', s)
        else:
            s = re.sub(filter_pattern, '', s)

    if keep_pattern:
        if isinstance(keep_pattern, (list, tuple)):
            for p in keep_pattern:
                s = ''.join(re.findall(p, s))
        else:
            s = ''.join(re.findall(keep_pattern, s))

    return s


def segments_from_paragraphs(paragraphs: List[str], sep=None, **filter_kwargs) -> List[List[str]]:
    """
    Usage:
        >>> segments_from_paragraphs(['hello world!'])
        [['hello', 'world!']]

        >>> segments_from_paragraphs(['hello world!'], filter_blank=True)
        [['helloworld!']]

        >>> from utils.excluded.charset_dict import utf8_pattern_dict
        >>> segments_from_paragraphs(['hello world!'], filter_pattern=utf8_pattern_dict['en_pr'])
        [['hello', 'world']]

        >>> segments_from_paragraphs(['hello world!'], keep_pattern=utf8_pattern_dict['en'])
        [['helloworld']]
    """
    segments = []
    for line in paragraphs:
        line = filter_func(line, **filter_kwargs)
        if sep == '':
            segments.append(list(line))
        else:
            segments.append(line.split(sep))

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

    paragraphs = map(lambda x: filter_func(x, **filter_kwargs), paragraphs)
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

