import re
from typing import List


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


def simple_cut(segments: List[str], sep=None, **filter_kwargs):
    """
    Usage:
        >>> simple_cut(['hello world!'])
        [['hello', 'world!']]

        >>> simple_cut(['hello world!'], filter_blank=True)
        [['helloworld!']]

        >>> from utils.excluded.charset_dict import utf8_pattern_dict
        >>> simple_cut(['hello world!'], filter_pattern=utf8_pattern_dict['en_pr'])
        [['hello', 'world']]

        >>> simple_cut(['hello world!'], keep_pattern=utf8_pattern_dict['en'])
        [['helloworld']]
    """
    _segments = []
    for line in segments:
        line = filter_func(line, **filter_kwargs)
        _segments.append(line.split(sep))

    return _segments


def jieba_cut(segments: List[str], **filter_kwargs) -> List[List[str]]:
    """
    Usage:
        >>> jieba_cut(['你好 世界！'])
        [['你好', ' ', '世界', '！']]

        >>> jieba_cut(['你好 世界！'], filter_blank=True)
        [['你好', '世界', '！']]

        >>> from utils.excluded.charset_dict import utf8_pattern_dict
        >>> jieba_cut(['你好 世界！'], filter_pattern=(utf8_pattern_dict['cjk_pr'], utf8_pattern_dict['en_pr_double']))
        [['你好', ' ', '世界']]

        >>> jieba_cut(['你好 世界！'], keep_pattern=utf8_pattern_dict['zh'])
        [['你好', '世界']]
    """
    import jieba

    segments = map(lambda x: filter_func(x, **filter_kwargs), segments)
    segments = map(jieba.lcut, segments)

    return list(segments)
