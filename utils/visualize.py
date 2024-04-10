import re
import cv2
import numpy as np
import inspect
from PIL import ImageFont, ImageDraw, Image
from typing import List
from .excluded.cmap import cmap, terminal_cmap

cmap_list = list(cmap.keys())
POLYGON = 1
RECTANGLE = 2


def get_color_array(idx):
    color_array = list(cmap[cmap_list[idx]]['array'])
    color_array[0], color_array[2] = color_array[2], color_array[0]  # rgb to bgr
    return tuple(color_array)


class ImageVisualize:
    @staticmethod
    def box(img, boxes, visual_type=RECTANGLE, colors=None, line_thickness=None):
        """目标框
        boxes: polygon: (-1, -1, 2) or rectangle: (-1, 4)
        colors: (-1, 3) or (-1, 1)
        """
        img = img.copy()
        colors = colors or [get_color_array(0)] * len(boxes)
        line_thickness = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

        for i in range(len(boxes)):
            if visual_type == POLYGON:  # polygon: (-1, -1, 2)
                cv2.polylines(img, [np.array(boxes[i], dtype=int)], isClosed=True, color=colors[i], thickness=line_thickness,
                              lineType=cv2.LINE_AA)

            elif visual_type == RECTANGLE:  # rectangle: (-1, 4)
                xyxy = boxes[i]
                c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(img, c1, c2, color=colors[i], thickness=line_thickness, lineType=cv2.LINE_AA)

            else:
                raise ValueError

        return img

    @staticmethod
    def text_box(img, text_boxes, texts, scores=None, drop_score=0.5, colors=None, font_path="utils/excluded/simfang.ttf"):
        """目标框 + 文本
        use PIL.Image instead of opencv for better chinese font support
        text_boxes: (-1, -1, 2)
        """
        scores = scores if scores is not None else [1] * len(text_boxes)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        h, w = image.height, image.width
        img_left = image.copy()
        img_right = Image.new('RGB', (w, h), (255, 255, 255))

        draw_left = ImageDraw.Draw(img_left)
        draw_right = ImageDraw.Draw(img_right)

        colors = colors or [get_color_array(0)] * len(text_boxes)

        for idx, (box, txt, score) in enumerate(zip(text_boxes, texts, scores)):
            if score < drop_score:
                continue

            color = colors[idx]

            box = [tuple(i) for i in box]
            draw_left.polygon(box, fill=color)
            draw_right.polygon(
                [
                    box[0][0], box[0][1], box[1][0], box[1][1], box[2][0],
                    box[2][1], box[3][0], box[3][1]
                ],
                outline=color)

            box_height = np.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
            box_width = np.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)

            # draw the text
            if box_height > 2 * box_width:
                font_size = max(int(box_width * 0.9), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                cur_x, cur_y = box[0][0] + 3, box[0][1]
                for c in txt:
                    char_size = font.getsize(c)
                    draw_right.text((cur_x, cur_y), c, fill=(0, 0, 0), font=font)
                    cur_y += char_size[1]
            else:
                font_size = max(int(box_height * 0.8), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                draw_right.text((box[0][0], box[0][1]), txt, fill=(0, 0, 0), font=font)

        img_left = Image.blend(image, img_left, 0.5)
        img_show = Image.new('RGB', (w * 2, h), (255, 255, 255))
        img_show.paste(img_left, (0, 0, w, h))
        img_show.paste(img_right, (w, 0, w * 2, h))

        draw_img = cv2.cvtColor(np.array(img_show), cv2.COLOR_RGB2BGR)

        return draw_img

    @staticmethod
    def text(img, text_boxes, texts, scores=None, drop_score=0.5, font_path="utils/excluded/simfang.ttf"):
        """文本
        use PIL.Image instead of opencv for better chinese font support
        text_boxes: (-1, 4, 2)
        """
        scores = scores if scores is not None else [1] * len(text_boxes)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        draw_image = ImageDraw.Draw(img)

        for idx, (box, txt, score) in enumerate(zip(text_boxes, texts, scores)):
            if score < drop_score:
                continue

            box = np.array(box)
            if box.size == 0:
                continue

            box_height = np.sqrt((box[0][0] - box[3][0]) ** 2 + (box[0][1] - box[3][1]) ** 2)
            box_width = np.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)

            # draw the text
            if box_height > 2 * box_width:
                font_size = min(max(int(box_width * 0.9), 10), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                cur_x, cur_y = box[0][0] + 3, box[0][1]

                for c in txt:
                    char_size = font.getsize(c)
                    draw_image.text((cur_x, cur_y), c, fill=(0, 0, 0), font=font)
                    cur_y += char_size[1]

            else:
                font_size = min(max(int(box_height * 0.8), 10), 10)
                n_font_per_line = max(int(box_width / font_size), 20)
                _txt = ''
                for i in range(0, len(txt), n_font_per_line):
                    _txt += txt[i:i + n_font_per_line] + '\n'
                txt = _txt[:-1]

                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                draw_image.text((box[0][0], box[0][1]), txt, fill=(0, 0, 0), font=font)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    @classmethod
    def label_box(cls, img, boxes, labels, colors=None, line_thickness=None):
        """目标框 + 标签
        boxes: (-1, 4)
        """
        img = img.copy()
        line_thickness = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        colors = colors or [get_color_array(0)] * len(boxes)
        labels = [str(i) for i in labels]

        img = cls.box(img, boxes, visual_type=RECTANGLE, colors=colors, line_thickness=line_thickness)

        # visual label
        for i in range(len(labels)):
            xyxy = boxes[i]
            c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

            tf = max(line_thickness - 1, 1)  # font thickness
            t_size = cv2.getTextSize(labels[i], 0, fontScale=line_thickness / 5, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, colors[i], -1, cv2.LINE_AA)  # filled
            cv2.putText(img, labels[i], (c1[0], c1[1] - 2), 0, line_thickness / 5, [225, 255, 255], thickness=tf,
                        lineType=cv2.LINE_AA)

        return img

    @staticmethod
    def block(img, boxes, visual_type=RECTANGLE, colors=None, alpha=1):
        """目标块
        boxes: polygon: (-1, -1, 2) or rectangle: (-1, 4)
        alpha: [0, 1], 1 gives opaque totally
        """
        img = img.copy()
        colors = colors or [get_color_array(0)] * len(boxes)
        boxes = np.array(boxes).astype(int)

        for i in range(len(boxes)):
            if visual_type == POLYGON:  # polygon: (-1, -1, 2)
                cv2.fillPoly(img, [np.array(boxes[i], dtype=int)], color=colors[i], lineType=cv2.LINE_AA)

            elif visual_type == RECTANGLE:  # rectangle: (-1, 4)
                x1, y1, x2, y2 = boxes[i]
                block = img[y1:y2, x1:x2]
                img[y1:y2, x1:x2] = (block * (1 - alpha) + (np.zeros_like(block) + colors[i]) * alpha).astype(img.dtype)

            else:
                raise ValueError

        return img


def get_variable_name(var):
    # there may be some bugs in the future
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


class TextVisualize:
    @staticmethod
    def get_start(types=None):
        if not types:
            return ''

        types = [types] if isinstance(types, str) else types

        # fmt -> \033[%sm
        start = '\033['
        for t in types:
            start += terminal_cmap[t] + ';'

        start = start[:-1] + 'm'

        return start

    @staticmethod
    def get_end(types=None):
        if not types:
            return ''
        return '\033[' + terminal_cmap['end'] + 'm'

    @classmethod
    def highlight_str(cls, text, types=None, fmt='', start='', end='', return_list=False, **kwargs):
        """hightlight a string

        Args:
            text(str or tuple):
                apply for `fmt % text`
            types(str or tuple):
                one of keys of `terminal_cmap',
                unuse when start and end is set,
                if fmt is not set, default is `('blue', 'bold')`
            fmt(str): highlight format, fmt like '<left>%s<right>'
            start(object):
            end(object)
            return_list(bool)

        Examples:
            >>> TextVisualize.highlight_str('hello')
            >>> TextVisualize.highlight_str('hello', 'blue')
            >>> TextVisualize.highlight_str('hello', ('blue', 'bold'))
            >>> TextVisualize.highlight_str('hello', fmt='<p style="color:red;">%s</p>')    # html type
            >>> TextVisualize.highlight_str('hello', 'blue', fmt='(highlight str: %s)')  # add special text

        """
        if not (types or fmt):
            types = ('blue', 'bold')

        if types:
            start = start or cls.get_start(types)
            end = end or cls.get_end(types)

        if not fmt:
            fmt = '%s'

        fmt = fmt % text
        s = [start, fmt, end]
        return s if return_list else ''.join(s)

    @classmethod
    def highlight_subtext(cls, text, span, highlight_obj=None,
                          keep_len=None, left_abbr='...', right_abbr='...',
                          auto_truncate=False, truncate_pattern=None,
                          return_list=False, **kwargs):
        """highlight a string where giving by span of a text
        See Also `TextVisualize.highlight_str`

        Args:
            text(str):
            span(tuple):
            highlight_obj(List[str]):
            keep_len(int):
                limit output str length, the len gives the length of left and right str.
                No limit if None, or exceeding part collapse to abbr str
            left_abbr(str)
            right_abbr(str)
            auto_truncate(bool): if true, truncate the text
            truncate_pattern(re.Patern)
            return_list(bool)
            kwargs: see also `TextVisualize.highlight_str()` to get more info

        Examples:
            >>> TextVisualize.highlight_subtext('123,4567,890abc', (5, 7))
            >>> TextVisualize.highlight_subtext('123,4567,890abc', (5, 7), keep_len=5)
            >>> TextVisualize.highlight_subtext('123,4567,890abc', (5, 7), keep_len=5, auto_truncate=True)

        """
        if not highlight_obj:
            highlight_obj = cls.highlight_str(text[span[0]:span[1]], return_list=return_list, **kwargs)
        highlight_obj = highlight_obj if return_list else [highlight_obj]

        if keep_len:
            left = max(0, span[0] - keep_len)
            right = min(len(text), span[1] + keep_len)

            if auto_truncate:
                truncate_pattern = truncate_pattern or re.compile(r'[。\.!\?！？;；,，]')
                a = text[left:right]
                r = truncate_pattern.split(a)
                if len(r) >= 3:  # make sure that returning one sentence at least
                    if left > 0 and not truncate_pattern.match(text[left - 1]):
                        _ = left + len(r[0]) + 1
                        if _ < span[0]:
                            left = _

                    if right < len(text) - 1 and not truncate_pattern.match(text[right + 1]):
                        _ = right - len(r[-1])
                        if _ > span[1]:
                            right = _

            left_abbr = left_abbr if left > 0 else ''
            right_abbr = right_abbr if right < len(text) else ''

            s = [
                left_abbr,
                text[left:span[0]],
                *highlight_obj,
                text[span[1]:right],
                right_abbr,
            ]

        else:
            s = [
                text[:span[0]],
                *highlight_obj,
                text[span[1]:]
            ]

        return s if return_list else ''.join(s)

    @classmethod
    def highlight_subtexts(cls, text, spans, highlight_objs=None, fmt='', return_list=False, ignore_overlap=False, **kwargs):
        """highlight multiple strings where giving by spans of a text
        See Also `TextVisualize.highlight_str`

        Args:
            text(str):
            spans(List[tuple]):
            highlight_objs(List[List[str]):
            fmt(str or list):
            return_list(bool):
            ignore_overlap(bool):
            kwargs: see also `TextVisualize.highlight_str()` to get more info

        Examples:
            >>> TextVisualize.highlight_subtexts('hello world', [(2, 3), (6, 7)])

        """
        if not spans:
            return text

        arg = np.argsort(spans, axis=0)

        s = []
        a = 0
        for i in arg[:, 0]:
            span = spans[i]
            if a > span[0]:
                if ignore_overlap:
                    print(f'{span = } overlap, please check')
                    continue
                else:
                    raise f'{span = } overlap, please check'

            if highlight_objs:
                highlight_obj = highlight_objs[i]
            else:
                _fmt = fmt[i] if isinstance(fmt, list) else fmt
                highlight_obj = cls.highlight_str(text[span[0]:span[1]], fmt=_fmt, return_list=return_list, **kwargs)

            highlight_obj = highlight_obj if return_list else [highlight_obj]
            s += [text[a:span[0]]] + highlight_obj
            a = span[1]

        s.append(text[a:])

        return s if return_list else ''.join(s)

    @classmethod
    def mark_subtext(cls, text, span, mark, types=('blue', 'bold'), fmt='', **kwargs):
        """highlight a string with mark symbols

        Examples:
            >>> TextVisualize.mark_subtext('hello', (2, 4), 'ii')
            >>> TextVisualize.mark_subtext('hello', (2, 4), 'ii', fmt='%s(to %s)')
        """
        fmt = fmt or '(%s -> %s)'
        highlight_obj = cls.highlight_str((text[span[0]:span[1]], mark), types=types, fmt=fmt, **kwargs)
        return cls.highlight_subtext(text, span, highlight_obj, **kwargs)

    @classmethod
    def mark_subtexts(cls, text, spans, marks, types=('blue', 'bold'), fmt='', **kwargs):
        """
        Examples:
            >>> TextVisualize.mark_subtexts('hello world', [(2, 3), (6, 7)], ['i', 'v'])
            >>> TextVisualize.mark_subtexts('hello world', [(2, 3), (6, 7)], ['i', 'v'], fmt='%s(to %s)')
        """
        _fmt = []
        highlight_objs = []
        for i, (span, mark) in enumerate(zip(spans, marks)):
            _fmt = fmt[i] if isinstance(fmt, list) else fmt
            _fmt = _fmt or '(%s -> %s)'
            highlight_obj = cls.highlight_str((text[span[0]:span[1]], mark), types=types, fmt=_fmt, **kwargs)
            highlight_objs.append(highlight_obj)
        return cls.highlight_subtexts(text, spans, highlight_objs, **kwargs)

    @staticmethod
    def num_to_human_readable_str(num: int, factor=1024., suffixes=('b', 'K', 'M', 'G', 'T')):
        """
        Examples:
            >>> TextVisualize.num_to_human_readable_str(1234567)
            1.18 M
            >>> TextVisualize.num_to_human_readable_str(1234567, factor=1e3)
            1.23 M
            >>> TextVisualize.num_to_human_readable_str(1234567, factor=(60., 60., 24.), suffixes=('s', 'm', 'h'))
            14.29 h
        """
        if not isinstance(factor, (list, tuple)):
            factor = [factor] * len(suffixes)

        for suffix, f in zip(suffixes, factor):
            if num >= f:
                num /= f
            else:
                return f'{num:.2f} {suffix}'

        return f'{num:.2f} {suffix}'

    @classmethod
    def dict_to_str(cls, dic: dict, return_list=False):
        """
        Examples:
            >>> TextVisualize.dict_to_str({'a': 1, 'b': {'c': 2, 'd': 3}})
            a=1,b.c=2,b.d=3
        """
        s = []
        for k, v in dic.items():
            if isinstance(v, dict):
                v = cls.dict_to_str(v, return_list=True)
                for vv in v:
                    s.append(f'{k}.{vv}')
            else:
                s.append(f'{k}={v}')
        return s if return_list else ','.join(s)
