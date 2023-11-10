import re


class FineGrainedSpliter:
    """article type split to different fine-grained type

    article (str): all lines fatten to a string
    paragraphs (List[List[str]]):
    chunks (List[List[str]]): each line has the same length possibly
    """

    def __init__(self, max_length=512, line_end='\n'):
        self.max_length = max_length
        self.line_end = line_end
        self.full_stop_rx = re.compile(r'.*(。|\.{3,6})(?![.。])', re.DOTALL)
        self.half_stop_rx = re.compile(r'.*[];；,，、》）}!?！？]', re.DOTALL)
        self.newline_stop_rx = re.compile(f'.+{line_end}', re.DOTALL)

    def gen_paragraphs(self, lines):
        for i, line in enumerate(lines):
            lines[i] = line + self.line_end

    def gen_article(self, paragraphs):
        return ''.join(paragraphs)

    def gen_chunks(self, paragraphs):
        chunks = []
        chunk = ''

        for p in paragraphs:
            chunk += p

            if len(chunk) > self.max_length:
                segs = self.segment_chunks(chunk)
                chunks.extend(segs[:-1])
                chunk = segs[-1]

        if chunk:
            chunks.append(chunk)

        return chunks

    def segment_chunks(self, text):
        segs = []
        rest = text

        while True:
            if len(rest) <= self.max_length:
                if rest:
                    segs.append(rest)
                break

            sect, rest = self.segment_one_chunk(rest)
            segs.append(sect)

        return segs

    def segment_one_chunk(self, text) -> tuple:
        if len(text) <= self.max_length:
            return text, ''

        tailing = text[self.max_length:]

        left_f, righ_f, is_matched_f = self.truncate_by_stop_symbol(text[:self.max_length], self.full_stop_rx)
        left_n, righ_n, is_matched_n = self.truncate_by_stop_symbol(text[:self.max_length], self.newline_stop_rx)

        if is_matched_f and is_matched_n:
            if len(left_f) >= len(left_n):
                return left_f, righ_f + tailing
            else:
                return left_n, righ_n + tailing
        elif is_matched_f:
            return left_f, righ_f + tailing
        elif is_matched_n:
            return left_n, righ_n + tailing

        left_h, righ_h, is_matched_h = self.truncate_by_stop_symbol(text[:self.max_length], self.half_stop_rx)
        if is_matched_h:
            return left_h, righ_h + tailing

        return text[:self.max_length], text[self.max_length:]

    def truncate_by_stop_symbol(self, text, pattern: re.Pattern) -> tuple:
        m = pattern.match(text)

        if m:
            left = text[:m.span()[1]]
            right = text[m.span()[1]:]
            is_matched = True
        else:
            left, right = text, ''
            is_matched = False

        return left, right, is_matched
