round_to_half_punctuation = {
    '（': '(',
    '）': ')',
    '【': '[',
    '】': ']',
    '；': ';',
    '：': ':',
    '“': '"',
    '”': '"',
    '‘': "'",
    '’': "'",
    '。': '.',
    '．': '.',
    '，': ',',
    '《': '<',
    '》': '>',
    '？': '?',
    '＜': '<',
    '＞': '>',
    '％': '%'
}

half_to_round_punctuation = {v: k for k, v in round_to_half_punctuation.items()}
