import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from typing import List

POLYGON = 1
RECTANGLE = 2


class ImageVisualize:
    @staticmethod
    def box(img, boxes, visual_type=RECTANGLE, colors=None, line_thickness=None):
        """添加若干个线框
        char_boxes: polygon: (-1, -1, 2) or rectangle: (-1, 4)
        """
        colors = colors or [tuple(np.random.randint(0, 255) for _ in range(3))] * len(boxes)
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
    def text_box(img, text_boxes, texts, scores=None, drop_score=0.5, colors=None, font_path="./font/simfang.ttf"):
        """将每个已识别的文本框住
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

        colors = colors or [tuple(np.random.randint(0, 255) for _ in range(3)) for _ in range(len(text_boxes))]

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
    def text(img, text_boxes, texts, scores=None, drop_score=0.5, font_path="./font/simfang.ttf"):
        """模拟文字识别效果
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
                font_size = max(int(box_width * 0.9), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                cur_x, cur_y = box[0][0] + 3, box[0][1]

                for c in txt:
                    char_size = font.getsize(c)
                    draw_image.text((cur_x, cur_y), c, fill=(0, 0, 0), font=font)
                    cur_y += char_size[1]

            else:
                font_size = max(int(box_height * 0.8), 10)
                font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
                draw_image.text((box[0][0], box[0][1]), txt, fill=(0, 0, 0), font=font)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    @classmethod
    def label_box(cls, img, boxes, labels, colors=None, line_thickness=None):
        """将每个已识别的结构框住，并添加标签
        boxes: (-1, 4)
        """
        img = img.copy()
        line_thickness = line_thickness or round(0.001 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        colors = colors or [tuple(np.random.randint(0, 255) for _ in range(3))] * len(boxes)
        labels = [str(i) for i in labels]

        img = cls.box(img, boxes, visual_type=RECTANGLE, colors=colors, line_thickness=line_thickness)

        # cv2.rectangle(img, c1, c2, colors, thickness=line_thickness, lineType=cv2.LINE_AA)

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


class TextVisualize:
    # refer to https://en.wikipedia.org/wiki/ANSI_escape_code
    types = {
        # front color
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'magenta': '35',
        'cyan': '36',
        'white': '37',

        # background color
        'bg_black': '30',
        'bg_red': '31',
        'bg_green': '32',
        'bg_yellow': '33',
        'bg_blue': '34',
        'bg_magenta': '35',
        'bg_cyan': '36',
        'bg_white': '37',

        # bright colors
        'bright_black': '90',
        'bright_red': '91',
        'bright_green': '92',
        'bright_yellow': '93',
        'bright_blue': '94',
        'bright_magenta': '95',
        'bright_cyan': '96',
        'bright_white': '97',

        # misc
        'end': '0',
        'bold': '1',
        'underline': '4',
        'blink': '5',
        'reverse': '6',
        'invisible': '7',

    }

    @classmethod
    def highlight_str(cls, text, types='blue', start='', end=''):
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
                start += cls.types[t] + ';'

            start = start[:-1] + 'm'

        end = end or '\033[' + cls.types['end'] + 'm'

        return start + text + end

    @classmethod
    def highlight_subtext(
            cls,
            text, span, wing_length=None,
            types='blue', start='', end=''
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
            types='blue', start='', end=''

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
