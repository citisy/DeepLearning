from collections import defaultdict


def word_id_dict(segments):
    """

    Args:
        segments (List[List[str]]):

    Returns:
        a dict like {word: id}
    """
    word_dict = dict()
    for line in segments:
        for word in line:
            if word not in word_dict:
                # start with 1, and reserve 0 for unknown word usually
                word_dict[word] = len(word_dict) + 1

    return word_dict


def word_count_dict(segments):
    """

    Args:
        segments (List[List[str]]):

    Returns:
        a dict like {word: count}
    """
    word_dict = defaultdict(int)
    for line in segments:
        for word in line:
            word_dict[word] += 1
    return word_dict
