import re
import unicodedata
from typing import List


class Apply:
    def __init__(self, *funcs):
        self.funcs = funcs

    def __call__(self, obj):
        for func in self.funcs:
            obj = func(obj)
        return obj


class Base:
    def from_article(self, article: str):
        return self.from_paragraph(article)

    def from_paragraph(self, paragraph: str):
        raise NotImplemented

    def from_paragraphs(self, paragraphs: List[str]):
        return map(self.from_paragraph, paragraphs)

    def from_segment(self, segment: List[str]):
        return self.from_paragraphs(segment)


class FilterBlank(Base):
    """
    Usage:
        >>> FilterBlank().from_paragraph('hello world!')
        'helloworld!'
    """
    def from_paragraph(self, paragraph: str):
        return ''.join(paragraph.split())


class FilterPattern(Base):
    """
    Usage:
        >>> from utils.excluded.charset_dict import utf8_pattern_dict
        >>> FilterPattern(utf8_pattern_dict['en_pr']).from_paragraph('hello world!')
        'hello world'

        >>> FilterPattern(utf8_pattern_dict['cjk_pr'], utf8_pattern_dict['en_pr_double']).from_paragraph('你好 世界！')
        '你好 世界'
    """
    def __init__(self, *pattern: str or re.Pattern):
        patterns = []
        for p in pattern:
            if isinstance(p, str):
                patterns.append(p)
            else:
                patterns.append(p.pattern)
        self.pattern = re.compile('|'.join(patterns))

    def from_paragraph(self, paragraph: str):
        return re.sub(self.pattern, '', paragraph)


class KeepPattern(Base):
    """
    Usage:
        >>> from utils.excluded.charset_dict import utf8_pattern_dict
        >>> KeepPattern(utf8_pattern_dict['en'], re.compile(' ')).from_paragraph('hello world!你好 世界！')
        'hello world '

        >>> KeepPattern(utf8_pattern_dict['zh']).from_paragraph('hello world!你好 世界！')
        '你好世界'
    """
    def __init__(self, *pattern: str or re.Pattern):
        patterns = []
        for p in pattern:
            if isinstance(p, str):
                patterns.append(p)
            else:
                patterns.append(p.pattern)
        self.pattern = re.compile('|'.join(patterns))

    def from_paragraph(self, paragraph: str):
        return ''.join(re.findall(self.pattern, paragraph))


class Lower(Base):
    def from_paragraph(self, paragraph: str):
        return paragraph.lower()


class StripAccents(Base):
    """
    Usage:
        >>> StripAccents().from_paragraph('ü')
        'u'
    """
    def from_paragraph(self, paragraph: str):
        paragraph = unicodedata.normalize("NFD", paragraph)
        output = []
        for char in paragraph:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)
