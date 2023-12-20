"""
label text with ids, use key pairs of (acid, ), (pid, pcid) or (cid, ccid) to position a char,
easy to search chars from different fine-grained text type

article (str): all lines fatten to a string
segments (List[List[str]]): all lines after cut
paragraphs (List[str]): all original lines, the lines contain the sep symbol
chunks (List[str]): each line has the same length possibly

acid: article char id
pid: paragraph line id
pcid: paragraph char id
cid: chunk line id
ccid: chunk char id
"""


def get_full_id(
        acid=None, pid_pcid=None, cid_ccid=None,
        acid2pid_pcid=None, acid2cid_ccid=None, pid_pcid2acid=None, cid_ccid2acid=None,
):
    """input one id pairs of article, paragraphs or chunks, output full id pairs of others
    if acid is given, acid2pid_pcid and acid2cid_ccid must be given
    if pid_pcid is given, pid_pcid2acid and acid2cid_ccid must be given
    if cid_ccid is given, cid_ccid2acid and acid2pid_pcid must be given
    """
    if acid:
        pid_pcid = acid2pid_pcid[acid]
        cid_ccid = acid2cid_ccid[acid]

    elif pid_pcid:
        acid = pid_pcid2acid[acid]
        cid_ccid = acid2cid_ccid[acid]

    elif cid_ccid:
        acid = cid_ccid2acid[cid_ccid]
        pid_pcid = acid2pid_pcid[acid]

    return dict(
        acid=acid,
        pid_pcid=pid_pcid,
        cid_ccid=cid_ccid
    )


def full_id_pairs(paragraphs, chunks):
    """
    Usages:
        >>> full_id_pairs(['abcdefghijklmn'], ['abcde', 'fghij', 'klmn'])
        {'acid2pid_pcid': {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4), 5: (0, 5), 6: (0, 6), 7: (0, 7), 8: (0, 8), 9: (0, 9), 10: (0, 10), 11: (0, 11), 12: (0, 12), 13: (0, 13)},
        'pid_pcid2acid': {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (0, 5): 5, (0, 6): 6, (0, 7): 7, (0, 8): 8, (0, 9): 9, (0, 10): 10, (0, 11): 11, (0, 12): 12, (0, 13): 13},
        'acid2cid_ccid': {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4), 5: (1, 0), 6: (1, 1), 7: (1, 2), 8: (1, 3), 9: (1, 4), 10: (2, 0), 11: (2, 1), 12: (2, 2), 13: (2, 3)},
        'cid_ccid2acid': {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (1, 0): 5, (1, 1): 6, (1, 2): 7, (1, 3): 8, (1, 4): 9, (2, 0): 10, (2, 1): 11, (2, 2): 12, (2, 3): 13},
        'pid_pcid2cid_ccid': {(0, 0): (0, 0), (0, 1): (0, 1), (0, 2): (0, 2), (0, 3): (0, 3), (0, 4): (0, 4), (0, 5): (1, 0), (0, 6): (1, 1), (0, 7): (1, 2), (0, 8): (1, 3), (0, 9): (1, 4), (0, 10): (2, 0), (0, 11): (2, 1), (0, 12): (2, 2), (0, 13): (2, 3)},
        'cid_ccid2pid_pcid': {(0, 0): (0, 0), (0, 1): (0, 1), (0, 2): (0, 2), (0, 3): (0, 3), (0, 4): (0, 4), (1, 0): (0, 5), (1, 1): (0, 6), (1, 2): (0, 7), (1, 3): (0, 8), (1, 4): (0, 9), (2, 0): (0, 10), (2, 1): (0, 11), (2, 2): (0, 12), (2, 3): (0, 13)}}
    """
    r = dict()
    r.update(article_paragraphs_id_pairs(paragraphs))
    r.update(article_chunks_id_pairs(chunks))
    r.update(paragraphs_chunks_id_pairs(paragraphs, chunks, r['acid2pid_pcid'], r['acid2cid_ccid']))

    return r


def article_paragraphs_id_pairs(paragraphs):
    """
    Usages:
        >>> article_paragraphs_id_pairs(['abcdefghijklmn'])
        {'acid2pid_pcid': {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4), 5: (0, 5), 6: (0, 6), 7: (0, 7), 8: (0, 8), 9: (0, 9), 10: (0, 10), 11: (0, 11), 12: (0, 12), 13: (0, 13)},
        'pid_pcid2acid': {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (0, 5): 5, (0, 6): 6, (0, 7): 7, (0, 8): 8, (0, 9): 9, (0, 10): 10, (0, 11): 11, (0, 12): 12, (0, 13): 13}}
    """
    acid2pid_pcid = _gen(paragraphs)
    pid_pcid2acid = {v: k for k, v in acid2pid_pcid.items()}

    return dict(
        acid2pid_pcid=acid2pid_pcid,
        pid_pcid2acid=pid_pcid2acid
    )


def article_chunks_id_pairs(chunks):
    """
    Usages:
        >>> article_chunks_id_pairs(['abcde', 'fghij', 'klmn'])
        {'acid2cid_ccid': {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (0, 3), 4: (0, 4), 5: (1, 0), 6: (1, 1), 7: (1, 2), 8: (1, 3), 9: (1, 4), 10: (2, 0), 11: (2, 1), 12: (2, 2), 13: (2, 3)},
        'cid_ccid2acid': {(0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3, (0, 4): 4, (1, 0): 5, (1, 1): 6, (1, 2): 7, (1, 3): 8, (1, 4): 9, (2, 0): 10, (2, 1): 11, (2, 2): 12, (2, 3): 13}}
    """
    acid2cid_ccid = _gen(chunks)
    cid_ccid2acid = {v: k for k, v in acid2cid_ccid.items()}

    return dict(
        acid2cid_ccid=acid2cid_ccid,
        cid_ccid2acid=cid_ccid2acid
    )


def paragraphs_chunks_id_pairs(paragraphs, chunks, acid2pid_pcid=None, acid2cid_ccid=None):
    """
    Usages:
        >>> paragraphs_chunks_id_pairs(['abcdefghijklmn'], ['abcde', 'fghij', 'klmn'])
        {'pid_pcid2cid_ccid': {(0, 0): (0, 0), (0, 1): (0, 1), (0, 2): (0, 2), (0, 3): (0, 3), (0, 4): (0, 4), (0, 5): (1, 0), (0, 6): (1, 1), (0, 7): (1, 2), (0, 8): (1, 3), (0, 9): (1, 4), (0, 10): (2, 0), (0, 11): (2, 1), (0, 12): (2, 2), (0, 13): (2, 3)},
        'cid_ccid2pid_pcid': {(0, 0): (0, 0), (0, 1): (0, 1), (0, 2): (0, 2), (0, 3): (0, 3), (0, 4): (0, 4), (1, 0): (0, 5), (1, 1): (0, 6), (1, 2): (0, 7), (1, 3): (0, 8), (1, 4): (0, 9), (2, 0): (0, 10), (2, 1): (0, 11), (2, 2): (0, 12), (2, 3): (0, 13)}}
    """
    if acid2pid_pcid is None:
        r = article_paragraphs_id_pairs(paragraphs)
        acid2pid_pcid = r['acid2pid_pcid']

    if acid2cid_ccid is None:
        r = article_chunks_id_pairs(chunks)
        acid2cid_ccid = r['acid2cid_ccid']

    pid_pcid2cid_ccid = {}
    cid_ccid2pid_pcid = {}
    for k in acid2pid_pcid:
        pid_pcid = acid2pid_pcid[k]
        cid_ccid = acid2cid_ccid[k]
        pid_pcid2cid_ccid[pid_pcid] = cid_ccid
        cid_ccid2pid_pcid[cid_ccid] = pid_pcid

    return dict(
        pid_pcid2cid_ccid=pid_pcid2cid_ccid,
        cid_ccid2pid_pcid=cid_ccid2pid_pcid
    )


def _gen(lines):
    id_dic = {}
    base_id = 0
    for line_id, p in enumerate(lines):
        for char_id, _ in enumerate(p):
            id_dic[base_id] = (line_id, char_id)
            base_id += 1

    return id_dic
