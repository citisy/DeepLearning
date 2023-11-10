def num(s):
    pass


def round_to_half_punctuation(s: str):
    for a, b in convert_dict.round_to_half_punctuation.items():
        s = s.replace(a, b)
    return s


def _convert(s: str, d: dict):
    for a, b in d.items():
        s = s.replace(a, b)
    return s
