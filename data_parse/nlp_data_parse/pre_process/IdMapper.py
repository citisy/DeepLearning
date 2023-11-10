"""
label text with ids, use key pairs of (acid, ), (pid, pcid) or (cid, ccid) to position a char,
easy to search chars from different fine-grained text type

article (str): all lines fatten to a string
paragraphs (List[List[str]]):
chunks (List[List[str]]): each line has the same length possibly

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
    """input one id pairs of article, paragraphs or chunks, output full id pairs of others"""
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
    r = dict()
    r.update(article_paragraphs_id_pairs(paragraphs))
    r.update(article_chunks_id_pairs(chunks))
    r.update(paragraphs_chunks_id_pairs(paragraphs, chunks, r['acid2pid_pcid'], r['acid2cid_ccid']))

    return r


def article_paragraphs_id_pairs(paragraphs):
    acid2pid_pcid = _gen(paragraphs)
    pid_pcid2acid = {v: k for k, v in acid2pid_pcid.items()}

    return dict(
        acid2pid_pcid=acid2pid_pcid,
        pid_pcid2acid=pid_pcid2acid
    )


def article_chunks_id_pairs(chunks):
    acid2cid_ccid = _gen(chunks)
    cid_ccid2acid = {v: k for k, v in acid2cid_ccid.items()}

    return dict(
        acid2cid_ccid=acid2cid_ccid,
        cid_ccid2acid=cid_ccid2acid
    )


def paragraphs_chunks_id_pairs(paragraphs, chunks, acid2pid_pcid=None, acid2cid_ccid=None):
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
