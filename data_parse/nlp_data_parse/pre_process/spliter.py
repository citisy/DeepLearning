"""context split to different fine-grained type

article (str): all lines fatten to a string
paragraphs (List[str]): all original lines, each itme in list is a str line
chunks (List[str]): each line has the same length as paragraphs as possibly, each itme in list is a str line
segments (List[List[str]]): all lines after cut, each itme in list is a cut word list
seg_chunks (List[List[str]]): each line has the same length as segments as possibly, each itme in list is a cut word list
"""
import re
import unicodedata
from typing import List
from tqdm import tqdm
from utils.excluded.charset_dict import utf8_int_dict
from utils.visualize import TextVisualize


def paragraphs_from_article(article: str, sep='\n', keep_sep=False) -> List[str]:
    if keep_sep:
        paragraphs = [line + sep for line in article.split(sep)]
    else:
        paragraphs = article.split(sep)

    return paragraphs


def article_from_paragraphs(paragraphs: List[str], sep='\n') -> str:
    return sep.join(paragraphs)


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


def segments_from_paragraphs(paragraphs: List[str], verbose=False, **kwargs) -> List[List[str]]:
    """
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
    if verbose:
        paragraphs = tqdm(paragraphs, desc=TextVisualize.highlight_str('Split paragraphs'))

    for line in paragraphs:
        seg = segment_from_line(line, **kwargs)
        segments.append(seg)

    return segments


def segment_from_line(
        line, sep=None, is_split_punctuation=True,
        is_word_piece=False, vocab=None, **filter_kwargs):
    """

    Args:
        line:
        sep:
        is_split_punctuation:
        is_word_piece:
        vocab: for `word_piece`
        **filter_kwargs: kwargs for `filter_text()`

    """
    line = filter_text(line, **filter_kwargs)
    if sep == '':
        seg = list(line)
    else:
        seg = line.split(sep)

    if is_split_punctuation:
        seg = split_punctuation(seg)

    if is_word_piece:
        seg = word_piece(seg, vocab)

    return seg


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


def chunks_from_paragraphs(paragraphs: List[str], max_length=512, min_length=None):
    """
    Usage:
        >>> chunks_from_paragraphs(['abcdefghijklmn'], max_length=5)
        ['abcde', 'fghij', 'klmn']

        >>> chunks_from_paragraphs(['abc', 'def', 'ghi', 'jk', 'lmn'], max_length=5)
        ['abcde', 'fghij', 'klmn']

        >>> chunks_from_paragraphs(['abc', 'def', 'ghi', 'jk', 'lmn'], max_length=5, min_length=3)
        ['abc', 'def', 'ghi', 'jklmn']

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
        elif min_length and len(chunk) >= min_length:
            chunks.append(chunk)
            chunk = ''
            continue

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


full_stop_tokens = set('。.!?！？')
half_stop_tokens = set('];；,，、》）}')
newline_stop_tokens = set('\n')


def seg_chunks_from_segments(segments: List[List[str]], max_length=512, min_length=None, verbose=False):
    """see also `chunks_from_paragraphs()`"""
    seg_chunks = []
    seg_chunk = []

    if verbose:
        segments = tqdm(segments, desc=TextVisualize.highlight_str('Chunk segments'))

    for p in segments:
        seg_chunk += p

        if len(seg_chunk) > max_length:
            _chunks = seg_chunks_from_segment(seg_chunk, max_length=max_length)
            seg_chunks.extend(_chunks[:-1])
            seg_chunk = _chunks[-1]
        elif min_length and len(seg_chunk) >= min_length:
            seg_chunks.append(seg_chunk)
            seg_chunk = []
            continue

    if seg_chunk:
        seg_chunks.append(seg_chunk)

    return seg_chunks


def seg_chunks_from_segment(segment, max_length=512):
    seg_chunks = []
    rest = segment

    while True:
        if len(rest) <= max_length:
            if rest:
                seg_chunks.append(rest)
            break

        keep = rest[:max_length]
        tail = rest[max_length:]

        left_f, right_f, is_matched_f = truncate_by_stop_token(keep, full_stop_tokens)
        left_n, right_n, is_matched_n = truncate_by_stop_token(keep, newline_stop_tokens)

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
            left_h, right_h, is_matched_h = truncate_by_stop_token(keep, half_stop_tokens)
            if is_matched_h:
                sect, rest = left_h, right_h + tail
            else:
                sect, rest = keep, tail

        seg_chunks.append(sect)

    return seg_chunks


def truncate_by_stop_token(segment, token: set) -> tuple:
    is_matched = False
    left, right = segment, []

    for i, s in enumerate(segment):
        if s in token:
            left = segment[:i + 1]
            right = segment[i + 1:]
            is_matched = True

    return left, right, is_matched
