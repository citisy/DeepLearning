"""see
- https://www.unicode.org/charts/
- https://www.unicode.org/Public/UCD/latest/ucd/NamesList.txt
to get more info for all unicode characters
"""
import re


def gen_charset_int_dict():
    dic = {}
    for k, v in charset_str_dict.items():
        if isinstance(v[0], tuple):
            tmp = []
            for vv in v:
                tmp.append((ord(vv[0]), ord(vv[1])))
            dic[k] = tuple(tmp)
        else:
            dic[k] = (ord(v[0]), ord(v[1]))
    return dic


def get_charset_pattern_dict():
    dic = {}
    for k, v in charset_int_dict.items():
        if isinstance(v[0], tuple):
            p = ''
            for vv in v:
                p += f'[\\u{vv[0]:04x}-\\u{vv[1]:04x}]' + '|'
            p = p[:-1]
        else:
            p = f'[\\u{v[0]:04x}-\\u{v[1]:04x}]'
        dic[k] = re.compile(p)

    return dic


charset_str_dict = dict(

    # https://www.cnblogs.com/straybirds/p/6392306.html to get more info for zh utf8 characters
    zh_utf8=('\u4e00', '\u9fa5'),
    zh_gbk=('\x00', '\xff'),
    zh_gb2312=('\xa1', '\xff'),

    jp_hi_utf8=('\u3040', '\u309f'),
    jp_ka_utf8=('\u30a0', '\u30ff'),

    cjk_utf8=('\u2e80', '\u9fff'),
    cjk_pr_utf8=(('\u3000', '\u303f'), ('\ufe30', '\ufe4f')),

    en_utf8=('\u0041', '\u007a'),
    en_upper_utf8=('\u0041', '\u005a'),
    en_lower_utf8=('\u0061', '\u007a'),
    en_pr_utf8=(('\u0021', '\u002f'), ('\u003a', '\u0040'), ('\u005b', '\u0060'), ('\u007b', '\u007e')),
    num_utf8=('\u0030', '\u0039'),

    # double/full-width symbols
    en_double_utf8=('\uff21', '\uff5a'),
    en_upper_double_utf8=('\uff21', '\uff3a'),
    en_lower_double_utf8=('\uff41', '\uff5a'),
    en_pr_double_utf8=(('\uff01', '\uff0f'), ('\uff1a', '\uff20'), ('\uff3b', '\uff40'), ('\uff5b', '\uff5e')),
    num_double_utf8=('\uff10', '\uff19'),

    en_ascii=('\x41', '\x7a'),
    en_upper_ascii=('\x41', '\x5a'),
    en_lower_ascii=('\x61', '\x7a'),
    en_pr_ascii=('\x5b', '\x60'),

)

charset_int_dict = gen_charset_int_dict()
charset_pattern_dict = get_charset_pattern_dict()
