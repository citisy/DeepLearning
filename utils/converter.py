import base64
import cv2
import hashlib
import numpy as np
from pathlib import Path
from .os_lib import IgnoreException, FakeIo
from distutils.util import strtobool


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

        wh = np.r_[wh, wh]

        convert_bbox = convert_func(bbox, convert_bbox) * wh

        if flag:
            convert_bbox = convert_bbox[0]

        return convert_bbox

    @classmethod
    def mid_xywh2top_xyxy(cls, bbox, wh=None, blow_up=True):
        """中心点xywh转换成顶点xyxy

        Args:
            bbox: xywh, xy, middle coordinate, wh, width and height of object
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xyxy(tuple): 左上右下顶点xy坐标
        """

        def convert_func(bbox, convert_bbox):
            convert_bbox[:, 0:2] = bbox[:, 0:2] - bbox[:, 2:4] / 2
            convert_bbox[:, 2:4] = bbox[:, 0:2] + bbox[:, 2:4] / 2
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @classmethod
    def top_xywh2top_xyxy(cls, bbox, wh=None, blow_up=True):
        """顶点xywh转换成顶点xyxy

        Args:
            bbox: xywh, xy, left top coordinate, wh, width and height of object
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xyxy(tuple): 左上右下顶点xy坐标
        """

        def convert_func(bbox, convert_bbox):
            convert_bbox[:, 0:2] = bbox[:, 0:2]
            convert_bbox[:, 2:4] = bbox[:, 0:2] + bbox[:, 2:4]
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @classmethod
    def top_xywh2mid_xywh(cls, bbox, wh=None, blow_up=True):
        """顶点xywh转中心点xywh

        Args:
            bbox: xywh, xy, left top coordinate, wh, width and height of object
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xywh(tuple): xy -> 中心点坐标, wh -> 目标宽高
        """

        def convert_func(bbox, convert_bbox):
            convert_bbox[:, 0:2] = bbox[:, 0:2] + bbox[:, 2:4] / 2
            convert_bbox[:, 2:4] = bbox[:, 2:4]
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @classmethod
    def top_xyxy2top_xywh(cls, bbox, wh=None, blow_up=True):
        """顶点xyxy转顶点xywh

        Args:
            bbox: xyxy, left top and right down
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xywh(tuple): xy -> 左上顶点坐标, wh -> 目标宽高
        """

        def convert_func(bbox, convert_bbox):
            convert_bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, 0:2]
            convert_bbox[:, 0:2] = bbox[:, 0:2]
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @classmethod
    def top_xyxy2mid_xywh(cls, bbox, wh=None, blow_up=True):
        """顶点xyxy转换成中心点xywh

        Args:
            bbox: xyxy, left top and right down
            wh(tuple): 原始图片宽高，如果传入，则根据blow_up进行转换
            blow_up(bool): 是否放大

        Returns:
            xywh(tuple): xy -> 中心点点坐标, wh -> 目标宽高
        """

        def convert_func(bbox, convert_bbox):
            convert_bbox[:, 0:2] = (bbox[:, 0:2] + bbox[:, 2:4]) / 2
            convert_bbox[:, 2:4] = bbox[:, 2:4] - bbox[:, 0:2]
            return convert_bbox

        return cls._call(bbox, wh, blow_up, convert_func)

    @staticmethod
    def box2rect(boxes) -> np.ndarray:
        """bbox(-1, 4, 2) convert to rec(-1, 4)"""
        boxes = np.array(boxes)

        if boxes.size == 0:
            return np.zeros((0, 4))

        rects = np.zeros((len(boxes), 4))
        rects[:, :2] = boxes[:, 0]
        rects[:, 2:] = boxes[:, 2]
        return rects

    @staticmethod
    def rect2box(rects) -> np.ndarray:
        """rec(-1, 4) convert to bbox(-1, 4, 2)"""
        rects = np.array(rects)

        if rects.size == 0:
            return np.zeros((0, 0, 2))

        boxes = np.zeros((len(rects), 4, 2))
        boxes[:, 0] = rects[:, :2]
        boxes[:, 1] = rects[:, (2, 1)]
        boxes[:, 2] = rects[:, 2:]
        boxes[:, 3] = rects[:, (0, 3)]
        return boxes


ignore_exception = IgnoreException(stdout_method=FakeIo())


class DataConvert:
    @classmethod
    def custom_to_constant(cls, obj):
        """convert a custom type(like np.ndarray) to a constant type(like int, float, str)
        apply for json output

        >>>DataConvert.custom_to_constant({0: [np.array([1, 2, 3]), np.array([4, 5, 6])]})
        {0: [[1, 2, 3], [4, 5, 6]]}

        """
        if isinstance(obj, dict):
            obj = cls.dict_to_constant(obj)
        elif isinstance(obj, list):
            obj = cls.list_to_constant(obj)
        elif isinstance(obj, np.ndarray) and obj.dtype == np.uint8:
            obj = cls.img_array_to_constant(obj)
        elif isinstance(obj, np.ndarray) and obj.dtype != np.uint8:
            obj = cls.np_to_constant(obj)

        return obj

    @classmethod
    def dict_to_constant(cls, obj: dict):
        for k, v in obj.items():
            obj[k] = cls.custom_to_constant(v)

        return obj

    @classmethod
    def list_to_constant(cls, obj: list):
        for i, e in enumerate(obj):
            obj[i] = cls.custom_to_constant(e)
        return obj

    @staticmethod
    def np_to_constant(obj: np.ndarray):
        if obj.size == 1:
            if isinstance(obj, (np.float32, np.float64)):
                obj = float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                obj = int(obj)
        else:
            obj = obj.tolist()

        return obj

    @classmethod
    def img_array_to_constant(cls, obj: np.ndarray) -> str:
        return cls.image_to_base64(obj)

    @classmethod
    def image_to_base64(cls, obj: np.ndarray) -> str:
        retval, buffer = cv2.imencode('.png', obj)
        return cls.bytes_to_base64(buffer)

    @classmethod
    def base64_to_image(cls, obj: str) -> np.ndarray:
        png_original = cls.base64_to_bytes(obj)
        image_buffer = np.frombuffer(png_original, dtype=np.uint8)
        image = cv2.imdecode(image_buffer, flags=1)
        return image

    @staticmethod
    def bytes_to_base64(obj: bytes) -> str:
        return str(base64.b64encode(obj), 'utf-8')

    @staticmethod
    def base64_to_bytes(obj: str) -> bytes:
        return base64.b64decode(obj)

    @classmethod
    def file_to_base64(cls, obj: str or Path):
        with open(obj, 'rb') as f:
            return cls.bytes_to_base64(f.read())

    @classmethod
    def str_value_to_constant(cls, obj: dict):
        """
        >>> DataConvert.str_value_to_constant({0: '1', 1: '1.0', 2: 'true', 3: 'abc'})
        {0: 1, 1: 1.0, 2: 1, 3: 'abc'}

        """
        for k, v in obj.items():
            if isinstance(v, dict):
                cls.str_value_to_constant(v)

            elif isinstance(v, str):
                obj[k] = cls.str_to_constant(v)
        return obj

    @classmethod
    def str_to_constant(cls, obj: str):
        """
        >>> DataConvert.str_to_constant('1')
        1
        >>> DataConvert.str_to_constant('1.0')
        1.0
        >>> DataConvert.str_to_constant('true')
        True
        >>> DataConvert.str_to_constant('abc')
        'abc'

        """
        for func in [cls.str_to_int, cls.str_to_float, cls.str_to_bool]:
            s = func(obj)
            if s is not None:
                return s

        return obj

    @staticmethod
    @ignore_exception.add_ignore(err_type=ValueError)
    def str_to_int(obj):
        return int(obj)

    @staticmethod
    @ignore_exception.add_ignore(err_type=ValueError)
    def str_to_float(obj):
        return float(obj)

    @staticmethod
    @ignore_exception.add_ignore(err_type=ValueError)
    def str_to_bool(obj):
        return strtobool(obj)

    @classmethod
    def obj_to_md5(cls, obj):
        if isinstance(obj, bytes):
            return cls.bytes_to_md5(obj)
        elif isinstance(obj, str):
            return cls.str_to_md5(obj)
        elif isinstance(obj, dict):
            return cls.dict_to_md5(obj)
        elif isinstance(obj, Path):
            return cls.file_to_md5(obj)
        else:
            return cls.str_to_md5(str(obj))

    @staticmethod
    def bytes_to_md5(obj: bytes):
        return hashlib.md5(obj).hexdigest()

    @classmethod
    def str_to_md5(cls, obj: str):
        return cls.bytes_to_md5(obj.encode(encoding='utf8'))

    @classmethod
    def dict_to_md5(cls, obj: dict, sort=False):
        if sort:
            sort_keys = sorted(obj.keys())
            obj = {k: obj[k] for k in sort_keys}
        return cls.str_to_md5(str(obj))

    @classmethod
    def file_to_md5(cls, obj: str or Path):
        with open(obj, 'rb') as f:
            return cls.bytes_to_md5(f.read())

