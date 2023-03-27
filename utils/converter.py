import base64
import cv2
import numpy as np
import torch


class DataConvert:
    @staticmethod
    def image_to_base64(image):
        retval, buffer = cv2.imencode('.png', image)
        png_as_text = base64.b64encode(buffer)
        return png_as_text.decode('utf-8')

    @staticmethod
    def base64_to_image(string):
        png_original = base64.b64decode(string)
        image_buffer = np.frombuffer(png_original, dtype=np.uint8)
        image = cv2.imdecode(image_buffer, flags=1)
        return image


class CoordinateConvert:
    @staticmethod
    def _call(bbox, wh, blow_up, convert_func):
        tmp_bbox = np.array(bbox)
        flag = len(tmp_bbox.shape) == 1

        bbox = np.array(bbox).reshape((-1, 4))
        convert_bbox = np.zeros_like(bbox)

        if wh is None:
            wh = (1, 1)

        wh = np.array(wh)

        if not blow_up:
            wh = 1 / wh

        convert_bbox = convert_func(bbox, convert_bbox, wh)

        if flag:
            convert_bbox = convert_bbox[0]

        return convert_bbox

    @classmethod
    def mid_xywh2top_xyxy(cls, bbox, wh=None, blow_up=True):
        """中心点xywh转换成顶点xyxy

        Args:
            bbox(tuple): xywh, xy, middle coordinate, wh, width and height of object
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xyxy(tuple): 左上右下顶点xy坐标
        """
        def convert_func(bbox, convert_bbox, wh):
            convert_bbox[:, 0:2] = (bbox[:, 0:2] - bbox[:, 2:4] / 2) * wh
            convert_bbox[:, 2:4] = (bbox[:, 0:2] + bbox[:, 2:4] / 2) * wh
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @classmethod
    def top_xywh2top_xyxy(cls, bbox, wh=None, blow_up=True):
        """顶点xywh转换成顶点xyxy

        Args:
            bbox(tuple): xywh, xy, left top coordinate, wh, width and height of object
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xyxy(tuple): 左上右下顶点xy坐标
        """

        def convert_func(bbox, convert_bbox, wh):
            convert_bbox[:, 0:2] = bbox[:, 0:2] * wh
            convert_bbox[:, 2:4] = (bbox[:, 0:2] + bbox[:, 2:4]) * wh
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @classmethod
    def top_xywh2mid_xywh(cls, bbox, wh=None, blow_up=True):
        """顶点xywh转中心点xywh

        Args:
            bbox(tuple): xywh, xy, left top coordinate, wh, width and height of object
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xywh(tuple): xy -> 中心点坐标, wh -> 目标宽高
        """
        def convert_func(bbox, convert_bbox, wh):
            convert_bbox[:, 0:2] = (bbox[:, 0:2] + bbox[:, 2:4] / 2) * wh
            convert_bbox[:, 2:4] = bbox[:, 2:4] * wh
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @classmethod
    def top_xyxy2top_xywh(cls, bbox, wh=None, blow_up=True):
        """顶点xyxy转顶点xywh

        Args:
            bbox(tuple): xyxy, left top and right down
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xywh(tuple): xy -> 左上顶点坐标, wh -> 目标宽高
        """
        def convert_func(bbox, convert_bbox, wh):
            convert_bbox[:, 2:4] = (bbox[:, 0:2] + bbox[:, 2:4]) / 2 * wh
            convert_bbox[:, 0:2] = bbox[:, 0:2] * wh
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @staticmethod
    def box2rect(boxes) -> np.ndarray:
        """(-1, 4, 2) -> 4 * (xy) change to (-1, 4) -> xyxy"""
        boxes = np.array(boxes)

        if boxes.size == 0:
            return np.array([[]])

        rects = np.zeros((len(boxes), 4))
        rects[:, :2] = boxes[:, 0]
        rects[:, 2:] = boxes[:, 2]
        return rects

    @staticmethod
    def rect2box(rects) -> np.ndarray:
        """(-1, 4) -> xyxy change to (-1, 4, 2) -> 4 * (xy)"""
        rects = np.array(rects)

        if rects.size == 0:
            return np.zeros((0, 0, 2))

        boxes = np.zeros((len(rects), 4, 2))
        boxes[:, 0] = rects[:, :2]
        boxes[:, 1] = rects[:, (2, 1)]
        boxes[:, 2] = rects[:, 2:]
        boxes[:, 3] = rects[:, (0, 3)]
        return boxes


class DataTypeConvert:
    """convert a custom type(like np.ndarray) to a constant type(like int, float, str)"""

    @classmethod
    def parse_data(cls, obj):
        if isinstance(obj, dict):
            obj = cls.parse_dict(obj)
        elif isinstance(obj, list):
            obj = cls.parse_list(obj)
        if isinstance(obj, np.ndarray) and obj.dtype == np.uint8:
            obj = cls.parse_img(obj)
        elif isinstance(obj, np.ndarray) and obj.dtype != np.uint8:
            obj = cls.parse_numpy_array(obj)

        return obj

    @classmethod
    def parse_dict(cls, obj: dict):
        for k, v in obj.items():
            obj[k] = cls.parse_data(v)

        return obj

    @classmethod
    def parse_list(cls, obj: list):
        for i, e in enumerate(obj):
            obj[i] = cls.parse_data(e)
        return obj

    @staticmethod
    def parse_numpy_array(obj: np.ndarray):
        if obj.size == 1:
            if isinstance(obj, (np.float32, np.float64)):
                obj = float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                obj = int(obj)
        else:
            obj = obj.tolist()

        return obj

    @staticmethod
    def parse_img(obj: np.ndarray):
        return DataConvert.image_to_base64(obj)


class ModelConvert:
    @staticmethod
    def torch2jit(model, trace_input):
        with torch.no_grad():
            model.eval()
            # warmup, make sure that the model is initialized right
            model(trace_input)
            jit_model = torch.jit.trace(model, trace_input)

        return jit_model
