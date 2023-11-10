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
    def highlight_str(text, types=('blue', 'bold'), start='', end=''):
        """hightlight a string

        Args:
            text(str):
            types(str or tuple): one of keys of `TextVisualize.types'
            start(str): if set, ignore types
            end(str):

        Examples:
            >>> TextVisualize.highlight_str('hello', 'blue')
            >>> TextVisualize.highlight_str('hello', ('blue', 'bold'))
            >>> TextVisualize.highlight_str('hello', start='<p style="color:red;">', end='</p>')    # html type
            >>> TextVisualize.highlight_str('hello', start='\033[34m(highlight str: ', end=')\033[0m')  # add special text

        """
        if not start:
            types = [types] if isinstance(types, str) else types

            # fmt -> \033[%sm
            start = '\033['
            for t in types:
                start += terminal_cmap[t] + ';'

            start = start[:-1] + 'm'

        end = end or '\033[' + terminal_cmap['end'] + 'm'

        return start + text + end

    @classmethod
    def highlight_subtext(
            cls,
            text, span, wing_length=None,
            types=('blue', 'bold'), start='', end=''
    ):
        """highlight a string where giving by span of a text
        See Also `TextVisualize.highlight_str`

        Args:
            text(str):
            span(tuple):
            wing_length(int): limit str output. No limit if None, else exceeding part collapse to '...'
            types(str or tuple):
            start(str):
            end(str):

        Examples:
            >>> TextVisualize.highlight_subtext('hello', (2, 4))

        """
        if wing_length:
            left = max(0, span[0] - wing_length)
            right = min(len(text), span[1] + wing_length)
            left_abbr = '...' if left != 0 else ''
            right_abbr = '...' if right != len(text) else ''
            s = (
                    left_abbr
                    + text[left:span[0]]
                    + cls.highlight_str(text[span[0]:span[1]], types, start, end)
                    + text[span[1]:right]
                    + right_abbr
            )

        else:
            s = text[:span[0]] + cls.highlight_str(text[span[0]:span[1]], types, start, end) + text[span[1]:]

        return s

    @classmethod
    def highlight_subtexts(
            cls,
            text, spans,
            types=('blue', 'bold'), start='', end=''

    ):
        """highlight multiple strings where giving by spans of a text
        See Also `TextVisualize.highlight_str`

        Args:
            text(str):
            spans(List[tuple]):
            types(str or tuple):
            start(str or list):
            end(str or list):

        Examples:
            >>> TextVisualize.highlight_subtexts('hello world', [(2, 3), (6, 7)])

        """

        arg = np.argsort(spans, axis=0)

        s = ''
        tmp = 0
        for i in arg[:, 0]:
            span = spans[i]
            assert tmp <= span[0], f'{span = } overlap, please check'

            _start = start[i] if isinstance(start, list) else start
            _end = end[i] if isinstance(end, list) else end

            s += text[tmp:span[0]] + cls.highlight_str(text[span[0]:span[1]], types, _start, _end)
            tmp = span[1]

        s += text[tmp:]

        return s

    @staticmethod
    def num_to_human_readable_str(num: int):
        for suffix in ['b', 'K', 'M', 'G', 'T']:
            if num >= 1024:
                num /= 1024.
            else:
                return f'{num:.2f} {suffix}'

    @staticmethod
    def dict_to_str(dic: dict):
        tmp = [f'{k}={v}' for k, v in dic.items()]
        return ','.join(tmp)
