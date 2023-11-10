import re


def gen_int_dict(str_dict):
    dic = {}
    for k, v in str_dict.items():
        if isinstance(v[0], tuple):
            tmp = []
            for vv in v:
                tmp.append((ord(vv[0]), ord(vv[1])))
            dic[k] = tuple(tmp)
        else:
            dic[k] = (ord(v[0]), ord(v[1]))
    return dic


def get_unicode_pattern_dict(int_dict):
    dic = {}
    for k, v in int_dict.items():
        if isinstance(v[0], tuple):
            p = ''
            for vv in v:
                p += f'[\\u{vv[0]:04x}-\\u{vv[1]:04x}]' + '|'
            p = p[:-1]
        else:
            p = f'[\\u{v[0]:04x}-\\u{v[1]:04x}]'
        dic[k] = re.compile(p)

    return dic


def get_simple_pattern_dict(int_dict):
    dic = {}
    for k, v in int_dict.items():
        if isinstance(v[0], tuple):
            p = ''
            for vv in v:
                p += f'[\\x{vv[0]:02x}-\\x{vv[1]:02x}]' + '|'
            p = p[:-1]
        else:
            p = f'[\\x{v[0]:02x}-\\x{v[1]:02x}]'
        dic[k] = re.compile(p)

    return dic


"""see
- https://www.unicode.org/charts/
- https://www.unicode.org/Public/UCD/latest/ucd/NamesList.txt
to get more info for all unicode characters
"""
utf8_str_dict = dict(

    # https://www.cnblogs.com/straybirds/p/6392306.html to get more info for zh utf8 characters
    zh=('\u4e00', '\u9fa5'),

    jp_hi=('\u3040', '\u309f'),
    jp_ka=('\u30a0', '\u30ff'),

    cjk=('\u2e80', '\u9fff'),
    cjk_pr=(('\u3000', '\u303f'), ('\ufe30', '\ufe4f')),

    en=('\u0041', '\u007a'),
    en_upper=('\u0041', '\u005a'),
    en_lower=('\u0061', '\u007a'),
    en_pr=(('\u0021', '\u002f'), ('\u003a', '\u0040'), ('\u005b', '\u0060'), ('\u007b', '\u007e')),  # en punctuation
    num=('\u0030', '\u0039'),

    # double/full-width symbols
    en_double=('\uff21', '\uff5a'),
    en_upper_double=('\uff21', '\uff3a'),
    en_lower_double=('\uff41', '\uff5a'),
    en_pr_double=(('\uff01', '\uff0f'), ('\uff1a', '\uff20'), ('\uff3b', '\uff40'), ('\uff5b', '\uff5e')),  # en punctuation
    num_double=('\uff10', '\uff19'),
)

gbk_str_dict = dict(
    zh_gbk=('\x00', '\xff'),
)

gb2312_str_dict = dict(
    zh_gb2312=('\xa1', '\xff'),
)

ascii_str_dict = dict(
    en_ascii=('\x41', '\x7a'),
    en_upper_ascii=('\x41', '\x5a'),
    en_lower_ascii=('\x61', '\x7a'),
    en_pr_ascii=('\x5b', '\x60'),  # en punctuation
)

utf8_int_dict = gen_int_dict(utf8_str_dict)
utf8_pattern_dict = get_unicode_pattern_dict(utf8_int_dict)

gbk_int_dict = gen_int_dict(gbk_str_dict)
gbk_pattern_dict = get_simple_pattern_dict(gbk_int_dict)

gb2312_int_dict = gen_int_dict(gb2312_str_dict)
gb2312_pattern_dict = get_simple_pattern_dict(gb2312_int_dict)

ascii_int_dict = gen_int_dict(ascii_str_dict)
ascii_pattern_dict = get_simple_pattern_dict(ascii_int_dict)
