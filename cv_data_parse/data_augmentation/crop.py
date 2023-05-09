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
CENTER = 2
RANDOM = 3
W, H, WH = 1, 2, 3


class Pad:
    """See Also `torchvision.transforms.Pad` or `albumentations.PadIfNeeded`

    Args:
        direction (int): {1, 2, 3}
        pad_type (int): {0, 1, 2, 3}
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

    def __init__(self, direction=WH, pad_type=CENTER, fill_type=0, fill=(114, 114, 114)):
        self.direction = direction
        self.pad_type = pad_type
        self.fill_type = fill_type
        self.fill = fill

    def __call__(self, image, dst, bboxes=None, **kwargs):
        """

        Args:
            image: (h, w, c) image
            dst: padding to destination size

        """
        h, w, c = image.shape

        if self.direction == W:
            pad_info = self.get_lr_params(dst, w)
            pad_left, pad_right = pad_info['l'], pad_info['r']
            pad_top, pad_down = 0, 0

        elif self.direction == H:
            pad_info = self.get_td_params(dst, h)
            pad_left, pad_right = 0, 0
            pad_top, pad_down = pad_info['t'], pad_info['d']

        elif self.direction == WH:
            pad_info = self.get_lr_params(dst, w)
            pad_info.update(self.get_td_params(dst, h))
            pad_left, pad_right = pad_info['l'], pad_info['r']
            pad_top, pad_down = pad_info['t'], pad_info['d']

        else:
            raise ValueError(f'dont support {self.direction = }')

        if bboxes is not None:
            bboxes = np.array(bboxes)
            shift = np.array([pad_left, pad_top, pad_left, pad_top])
            bboxes += shift

        return {
            'image': cv2.copyMakeBorder(
                image, pad_top, pad_down, pad_left, pad_right,
                borderType=fill_mode[self.fill_type],
                value=self.fill
            ),
            'bboxes': bboxes,
            'crop.Pad': pad_info
        }

    def get_lr_params(self, dst, w):
        if self.pad_type == LEFT:
            pad_left, pad_right = dst - w, 0
        elif self.pad_type == RIGHT:
            pad_left, pad_right = 0, dst - w
        elif self.pad_type == CENTER:
            pad_left = (dst - w) // 2
            pad_right = dst - w - pad_left
        elif self.pad_type == RANDOM:
            pad_left = np.random.randint(dst - w)
            pad_right = dst - w - pad_left
        else:
            raise ValueError(f'dont support {self.pad_type = }')

        return dict(
            l=pad_left,
            r=pad_right
        )

    def get_td_params(self, dst, h):
        if self.pad_type == TOP:
            pad_top, pad_down = dst - h, 0
        elif self.pad_type == DOWN:
            pad_top, pad_down = 0, dst - h
        elif self.pad_type == CENTER:
            pad_top = (dst - h) // 2
            pad_down = dst - h - pad_top
        elif self.pad_type == RANDOM:
            pad_top = np.random.randint(dst - h)
            pad_down = dst - h - pad_top
        else:
            raise ValueError(f'dont support {self.pad_type = }')

        return dict(
            t=pad_top,
            d=pad_down
        )

    @staticmethod
    def restore(ret):
        params = ret.get('crop.Pad')
        bboxes = ret['bboxes']

        pad_left = params.get('l', 0)
        pad_top = params.get('t', 0)

        bboxes = np.array(bboxes)
        shift = np.array([pad_left, pad_top, pad_left, pad_top])
        bboxes -= shift

        ret['bboxes'] = bboxes

        return ret


class Crop:
    def __init__(self, is_pad=True, **pad_kwargs):
        self.is_pad = is_pad
        self.w_pad = Pad(direction=W, **pad_kwargs)
        self.h_pad = Pad(direction=H, **pad_kwargs)

    def __call__(self, image, dst_coor, bboxes=None, classes=None, **kwargs):
        x1, x2, y1, y2 = dst_coor
        h, w, c = image.shape

        assert x1 >= 0 and y1 >= 0, ValueError(f'{x1 = } and {y1 = } must not be smaller than 0')

        ret = dict(image=image, bboxes=bboxes)

        if x2 > w or y2 > h:
            if self.is_pad:
                if x2 > w:
                    ret = merge_dict(ret, self.w_pad(dst=x2, **ret))
                if y2 > h:
                    ret = merge_dict(ret, self.h_pad(dst=y2, **ret))

            else:
                raise ValueError(f'image width = {w} and height = {h} must be greater than {x2 = } and {y2 = } or set pad=True')

        if bboxes is not None:
            bboxes = ret['bboxes']
            shift = np.array([x1, y1, x1, y1])
            bboxes -= shift
            bboxes = bboxes.clip(min=0)
            bboxes[:, 0::2] = np.where(bboxes[:, 0::2] > x2 - x1, x2 - x1, bboxes[:, 0::2])
            bboxes[:, 1::2] = np.where(bboxes[:, 1::2] > y2 - y1, y2 - y1, bboxes[:, 1::2])

            idx = ~((bboxes[:, 0] == bboxes[:, 2]) | (bboxes[:, 1] == bboxes[:, 3]))
            bboxes = bboxes[idx]

            if classes is not None:
                classes = classes[idx]

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
            'classes': classes
        })

        return ret

    def restore(self, ret):
        params = ret.get('crop.Crop')
        bboxes = ret['bboxes']

        x1 = params['l']
        y1 = params['t']

        shift = np.array([x1, y1, x1, y1])
        bboxes += shift

        ret['bboxes'] = bboxes

        return self.h_pad.restore(ret)


class Random:
    """See Also `torchvision.transforms.RandomCrop` or `albumentations.RandomCrop`"""

    def __init__(self, is_pad=True, **pad_kwargs):
        self.crop = Crop(is_pad=is_pad, **pad_kwargs)

    def __call__(self, image, dst, **kwargs):
        """(w, h) -> (dst, dst)"""
        h, w, c = image.shape

        w_ = np.random.randint(w - dst) if w > dst else 0
        h_ = np.random.randint(h - dst) if h > dst else 0

        return self.crop(image, (w_, w_ + dst, h_, h_ + dst), **kwargs)


class Corner:
    def __init__(self, pos=(LEFT, TOP), is_pad=True, **pad_kwargs):
        self.pos = pos
        self.crop = Crop(is_pad=is_pad, **pad_kwargs)

    def __call__(self, image, dst, **kwargs):
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

        return self.crop(image, (w_, w_ + dst, h_, h_ + dst), **kwargs)


class FiveCrop:
    """See Also `torchvision.transforms.FiveCrop`"""

    def __init__(self, is_pad=True, **pad_kwargs):
        funcs = []

        for pos in ((LEFT, TOP), (LEFT, DOWN), (RIGHT, TOP), (RIGHT, DOWN)):
            funcs.append(Corner(pos, is_pad, **pad_kwargs))

        funcs.append(Center(is_pad, **pad_kwargs))
        self.apply = Apply(funcs, full_result=True, replace=False)

    def __call__(self, image, dst, bboxes=None, classes=None, **kwargs):
        return self.apply(image=image, dst=dst, bboxes=bboxes, classes=classes)


class TenCrop:
    """See Also `torchvision.transforms.TenCrop`"""

    def __init__(self, is_pad=True, **pad_kwargs):
        funcs = []

        for pos in ((LEFT, TOP), (LEFT, DOWN), (RIGHT, TOP), (RIGHT, DOWN)):
            funcs.append(Corner(pos, is_pad, **pad_kwargs))

        funcs.append(Center(is_pad, **pad_kwargs))
        self.apply = Apply(funcs, full_result=True, replace=False)
        self.shift = geometry.VFlip()

    def __call__(self, image, dst, bboxes=None, classes=None, **kwargs):
        ret = self.apply(image=image, dst=dst, bboxes=bboxes)
        h, w = image.shape[:2]
        _ = self.shift(image, bboxes, (w, h))
        image, bboxes = _['image'], _['bboxes']

        _ret = self.apply(image=image, dst=dst, bboxes=bboxes, classes=classes)
        ret += _ret

        return ret


class Center:
    """See Also `torchvision.transforms.CenterCrop` or `albumentations.CenterCrop`"""

    def __init__(self, is_pad=True, **pad_kwargs):
        self.crop = Crop(is_pad=is_pad, **pad_kwargs)

    def __call__(self, image, dst, **kwargs):
        h, w, c = image.shape

        w_ = max(w - dst, 0) // 2
        h_ = max(h - dst, 0) // 2

        return self.crop(image, (w_, w_ + dst, h_, h_ + dst), **kwargs)

    def restore(self, ret):
        return self.crop.restore(ret)
