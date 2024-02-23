"""
label text with ids, use key pairs of (acid, ), (pid, pcid) or (cid, ccid) to position a char,
easy to search chars from different fine-grained text type

article (str): all lines fatten to a string
paragraphs (List[str]): all original lines, each itme in list is a str line
chunks (List[str]): each line has the same length as paragraphs as possibly, each itme in list is a str line
segments (List[List[str]]): all lines after cut, each itme in list is a cut word list
seg_chunks (List[List[str]]): each line has the same length as segments as possibly, each itme in list is a cut word list

acid: article char id
pid: paragraph line id
pcid: paragraph char id
cid: chunk line id
ccid: chunk char id
"""


class Mapper:
    acid2pid_pcid: dict
    pid_pcid2acid: dict
    acid2cid_ccid: dict
    cid_ccid2acid: dict
    pid_pcid2cid_ccid: dict
    cid_ccid2pid_pcid: dict

    def __init__(self, paragraphs=None, chunks=None):
        self.__dict__.update(full_id_pairs(paragraphs, chunks))

    def get_full_id(self, acid=None, pid_pcid=None, cid_ccid=None):
        """input one id pairs of article, paragraphs or chunks, output full id pairs of others
        if acid is given, acid2pid_pcid and acid2cid_ccid must be given
        if pid_pcid is given, pid_pcid2acid and acid2cid_ccid must be given
        if cid_ccid is given, cid_ccid2acid and acid2pid_pcid must be given
        """
        if acid:
            pid_pcid = self.acid2pid_pcid[acid]
            cid_ccid = self.acid2cid_ccid[acid]

        elif pid_pcid:
            acid = self.pid_pcid2acid[pid_pcid]
            cid_ccid = self.acid2cid_ccid[acid]

        elif cid_ccid:
            acid = self.cid_ccid2acid[cid_ccid]
            pid_pcid = self.acid2pid_pcid[acid]

        else:
            raise ValueError('Must input one of [acid, pid_pcid, cid_ccid]')

        return dict(
            acid=acid,
            pid_pcid=pid_pcid,
            cid_ccid=cid_ccid
        )

    def get_full_span(self, length, acid=None, pid_pcid=None, cid_ccid=None):
        if acid:
            start_acid = acid
            start_pid_pcid = self.acid2pid_pcid[start_acid]
            start_cid_ccid = self.acid2cid_ccid[start_acid]

            end_acid = acid + length
            end_pid_pcid = self.acid2pid_pcid[end_acid]
            end_cid_ccid = self.acid2cid_ccid[end_acid]

        elif pid_pcid:
            start_pid_pcid = pid_pcid
            start_acid = self.pid_pcid2acid[start_pid_pcid]
            start_cid_ccid = self.acid2cid_ccid[start_acid]

            end_pid_pcid = (pid_pcid[0], pid_pcid[1] + length)
            end_acid = self.pid_pcid2acid[end_pid_pcid]
            end_cid_ccid = self.acid2cid_ccid[end_acid]

        elif cid_ccid:
            start_cid_ccid = cid_ccid
            start_acid = self.cid_ccid2acid[start_cid_ccid]
            start_pid_pcid = self.acid2pid_pcid[start_acid]

            end_cid_ccid = (cid_ccid[0], cid_ccid[1] + length)
            end_acid = self.cid_ccid2acid[end_cid_ccid]
            end_pid_pcid = self.acid2pid_pcid[end_acid]

        else:
            raise ValueError('Must input one of [acid, pid_pcid, cid_ccid]')

        return dict(
            acid_span=(start_acid, end_acid),
            pid_pcid_span=(start_pid_pcid, end_pid_pcid),
            cid_ccid_span=(start_cid_ccid, end_cid_ccid)
        )


def full_id_pairs(paragraphs=None, chunks=None):
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
    if paragraphs:
        r.update(article_paragraphs_id_pairs(paragraphs))
    if chunks:
        r.update(article_chunks_id_pairs(chunks))
    if paragraphs and chunks:
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
