"""change the shape of image by cutting the image directly"""
import cv2
import numpy as np
from . import Apply, geometry
from utils.configs import merge_dict

fill_mode = [
    cv2.BORDER_CONSTANT,
    cv2.BORDER_REPLICATE,
    cv2.BORDER_REFLECT_101,
    cv2.BORDER_REFLECT,
    cv2.BORDER_WRAP
]

LEFT, RIGHT, TOP, DOWN = 0, 1, 0, 1


class Pad:
    """See Also `torchvision.transforms.Pad`

    Args:
        direction: w or h
        pad_type: {0, 1, 2, 3}
            if direction is w, 0 give left, 1 give right, 2 give left and right averagely, 3 give left and right randomly
            if direction is h, 0 give up, 1 give down, 2 give up and down averagely, 3 give up and down randomly
        fill_type: {0, 1, 2, 3, 4}
            0, pads with a constant value, this value is specified with fill
            1, pads with the last value on the edge of the image
            2, pads with reflection of image (without repeating the last value on the edge),
                e.g. pad 2 elements in the end: [1, 2, 3, 4] -> [1, 2, 3, 4, 3, 2]
            3, pads with reflection of image (repeating the last value on the edge)
                e.g. pad 2 elements in the end: [1, 2, 3, 4] -> [1, 2, 3, 4, 4, 3]
            4, pads with reflection of image (without rotating the last value on the edge)
                e.g. pad 2 elements in the end: [1, 2, 3, 4] -> [1, 2, 3, 4, 3, 4]
        fill (int or tuple): fill value

    """

    def __init__(self, direction='w', pad_type=2, fill_type=0, fill=(114, 114, 114)):
        self.direction = direction
        self.pad_type = pad_type
        self.fill_type = fill_type
        self.fill = fill

    def __call__(self, image, dst, **kwargs):
        """

        Args:
            image: (h, w, c) image
            dst: padding to destination size

        """
        h, w, c = image.shape
        ret = dict(image=image)

        if self.direction == 'w':
            if self.pad_type == 0:
                pad_left, pad_right = dst - w, 0
            elif self.pad_type == 1:
                pad_left, pad_right = 0, dst - w
            elif self.pad_type == 2:
                pad_left = (dst - w) // 2
                pad_right = dst - w - pad_left
            elif self.pad_type == 3:
                pad_left = np.random.randint(dst - w)
                pad_right = dst - w - pad_left
            else:
                raise ValueError(f'dont support {self.pad_type = }')

            pad_top, pad_down = 0, 0
            pad_info = dict(
                l=pad_left,
                r=pad_right
            )

        elif self.direction == 'h':
            if self.pad_type == 0:
                pad_top, pad_down = dst - h, 0
            elif self.pad_type == 1:
                pad_top, pad_down = 0, dst - h
            elif self.pad_type == 2:
                pad_top = (dst - h) // 2
                pad_down = dst - h - pad_top
            elif self.pad_type == 3:
                pad_top = np.random.randint(dst - h)
                pad_down = dst - h - pad_top
            else:
                raise ValueError(f'dont support {self.pad_type = }')

            pad_left, pad_right = 0, 0
            pad_info = dict(
                t=pad_top,
                d=pad_down
            )

        else:
            raise ValueError(f'dont support {self.direction = }')

        ret.update({
            'image': cv2.copyMakeBorder(
                image, pad_top, pad_down, pad_left, pad_right,
                borderType=fill_mode[self.fill_type],
                value=self.fill
            ),
            'crop.Pad': pad_info
        })

        return ret


class Crop:
    def __init__(self, is_pad=True, **pad_kwargs):
        self.is_pad = is_pad
        self.w_pad = Pad(direction='w', **pad_kwargs)
        self.h_pad = Pad(direction='h', **pad_kwargs)

    def __call__(self, image, dst_coor, bboxes=None, **kwargs):
        x1, x2, y1, y2 = dst_coor
        h, w, c = image.shape

        assert x1 >= 0 and y1 >= 0, ValueError(f'{x1 = } and {y1 = } must not be smaller than 0')

        ret = dict(image=image)

        if x2 > w or y2 > h:
            if self.is_pad:
                if x2 > w:
                    ret = merge_dict(ret, self.w_pad(ret['image'], x2))
                if y2 > h:
                    ret = merge_dict(ret, self.h_pad(ret['image'], y2))

            else:
                raise ValueError(f'image width = {w} and height = {h} must be greater than {x2 = } and {y2 = } or set pad=True')

        if bboxes is not None:
            bboxes = np.array(bboxes)
            if 'crop.Pad' in ret:
                pad_info = ret['crop.Pad']
                shift = np.array([
                    pad_info.get('l', 0),
                    pad_info.get('t', 0),
                    pad_info.get('l', 0),
                    pad_info.get('t', 0),
                ])
                bboxes += shift

            shift = np.array([x1, y1, x1, y1])
            bboxes -= shift
            bboxes = bboxes.clip(min=0)
            bboxes[:, 0::2] = np.where(bboxes[:, 0::2] > x2 - x1, x2 - x1, bboxes[:, 0::2])
            bboxes[:, 1::2] = np.where(bboxes[:, 1::2] > y2 - y1, y2 - y1, bboxes[:, 1::2])

        image = ret['image'][y1:y2, x1: x2]
        ret.update({
            'image': image,
            'crop.Crop': dict(
                l=x1,
                r=w - x2,
                t=y1,
                d=h - y2,
            ),
            'bboxes': bboxes,
        })

        return ret


class Random:
    def __init__(self, is_pad=True, **pad_kwargs):
        self.crop = Crop(is_pad=is_pad, **pad_kwargs)

    def __call__(self, image, dst, bboxes=None, **kwargs):
        """(w, h) -> (dst, dst)
        See Also `torchvision.transforms.RandomCrop`"""
        h, w, c = image.shape

        w_ = np.random.randint(w - dst) if w > dst else 0
        h_ = np.random.randint(h - dst) if h > dst else 0

        return self.crop(image, (w_, w_ + dst, h_, h_ + dst), bboxes)


class Corner:
    def __init__(self, pos=(LEFT, TOP), is_pad=True, **pad_kwargs):
        self.pos = pos
        self.crop = Crop(is_pad=is_pad, **pad_kwargs)

    def __call__(self, image, dst, bboxes=None, **kwargs):
        h, w, c = image.shape

        if self.pos == (LEFT, TOP):
            w_, h_ = 0, 0
        elif self.pos == (RIGHT, TOP):
            w_, h_ = 0, max(h - dst, 0)
        elif self.pos == (LEFT, DOWN):
            w_, h_ = max(w - dst, 0), 0
        elif self.pos == (RIGHT, DOWN):
            w_, h_ = max(w - dst, 0), max(h - dst, 0)
        else:
            raise ValueError(f'dont support {self.pos = }')

        return self.crop(image, (w_, w_ + dst, h_, h_ + dst), bboxes)


class FiveCrop:
    """See Also `torchvision.transforms.FiveCrop`"""

    def __init__(self, is_pad=True, **pad_kwargs):
        funcs = []

        for pos in ((LEFT, TOP), (LEFT, DOWN), (RIGHT, TOP), (RIGHT, DOWN)):
            funcs.append(Corner(pos, is_pad, **pad_kwargs))

        funcs.append(Center(is_pad, **pad_kwargs))
        self.apply = Apply(funcs, full_result=True, replace=False)

    def __call__(self, image, dst, bboxes=None, **kwargs):
        return self.apply(image=image, dst=dst, bboxes=bboxes)


class TenCrop:
    """See Also `torchvision.transforms.TenCrop`"""

    def __init__(self, is_pad=True, **pad_kwargs):
        funcs = []

        for pos in ((LEFT, TOP), (LEFT, DOWN), (RIGHT, TOP), (RIGHT, DOWN)):
            funcs.append(Corner(pos, is_pad, **pad_kwargs))

        funcs.append(Center(is_pad, **pad_kwargs))
        self.apply = Apply(funcs, full_result=True, replace=False)
        self.shift = geometry.VFlip()

    def __call__(self, image, dst, bboxes=None, **kwargs):
        ret = self.apply(image=image, dst=dst, bboxes=bboxes)
        h, w = image.shape[:2]
        _ = self.shift(image, bboxes, (w, h))
        image, bboxes = _['image'], _['bboxes']

        _ret = self.apply(image=image, dst=dst, bboxes=bboxes)
        ret += _ret

        return ret


class Center:
    def __init__(self, is_pad=True, **pad_kwargs):
        self.crop = Crop(is_pad=is_pad, **pad_kwargs)

    def __call__(self, image, dst, bboxes=None, **kwargs):
        """See Also `torchvision.transforms.CenterCrop`"""
        h, w, c = image.shape

        w_ = max(w - dst, 0) // 2
        h_ = max(h - dst, 0) // 2

        return self.crop(image, (w_, w_ + dst, h_, h_ + dst), bboxes)
