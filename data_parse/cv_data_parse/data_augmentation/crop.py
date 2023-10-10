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
        pad_type (int or tuple): {0, 1, 2, 3}
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

    def __init__(self, pad_type=CENTER, fill_type=0, fill=(114, 114, 114)):
        if isinstance(pad_type, int):
            pad_type = (pad_type, pad_type)
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.pad_type = pad_type
        self.fill_type = fill_type
        self.fill = fill

    def get_params(self, dst, w, h):
        if isinstance(dst, int):
            dst = (dst, dst)
        pad_info = {}
        if dst[0] > w:
            pad_info.update(self.get_lr_params(dst[0], w))
        if dst[1] > h:
            pad_info.update(self.get_td_params(dst[1], h))
        return pad_info

    def get_lr_params(self, dst, w):
        pad_type = self.pad_type[0]
        if pad_type == LEFT:
            pad_left, pad_right = dst - w, 0
        elif pad_type == RIGHT:
            pad_left, pad_right = 0, dst - w
        elif pad_type == CENTER:
            pad_left = (dst - w) // 2
            pad_right = dst - w - pad_left
        elif pad_type == RANDOM:
            pad_left = np.random.randint(dst - w)
            pad_right = dst - w - pad_left
        else:
            raise ValueError(f'dont support {pad_type = }')

        return dict(
            l=pad_left,
            r=pad_right
        )

    def get_td_params(self, dst, h):
        pad_type = self.pad_type[1]
        if pad_type == TOP:
            pad_top, pad_down = dst - h, 0
        elif pad_type == DOWN:
            pad_top, pad_down = 0, dst - h
        elif pad_type == CENTER:
            pad_top = (dst - h) // 2
            pad_down = dst - h - pad_top
        elif pad_type == RANDOM:
            pad_top = np.random.randint(dst - h)
            pad_down = dst - h - pad_top
        else:
            raise ValueError(f'dont support {pad_type = }')

        return dict(
            t=pad_top,
            d=pad_down
        )

    def get_add_params(self, dst, w, h):
        return {self.name: self.get_params(dst, w, h)}

    def parse_add_params(self, ret):
        pad_info = ret[self.name]
        t = pad_info.get('t', 0)
        d = pad_info.get('d', 0)
        l = pad_info.get('l', 0)
        r = pad_info.get('r', 0)
        return t, d, l, r

    def __call__(self, image, dst, bboxes=None, **kwargs):
        """

        Args:
            image: (h, w, c) image
            dst: padding to destination size

        """
        h, w, c = image.shape
        add_params = self.get_add_params(dst, w, h)
        image = self.apply_image(image, add_params)
        bboxes = self.apply_bboxes(bboxes, add_params)

        return {
            'image': image,
            'bboxes': bboxes,
            **add_params
        }

    def apply_image(self, image, ret):
        if self.name in ret:
            t, d, l, r = self.parse_add_params(ret)
            return cv2.copyMakeBorder(
                image, t, d, l, r,
                borderType=fill_mode[self.fill_type],
                value=self.fill,
            )
        else:
            return image

    def apply_bboxes(self, bboxes, ret):
        if bboxes is not None and self.name in ret:
            t, d, l, r = self.parse_add_params(ret)
            bboxes = np.array(bboxes)
            shift = np.array([l, t, l, t])
            bboxes += shift

        return bboxes

    def restore(self, ret):
        if self.name in ret:
            t, d, l, r = self.parse_add_params(ret)

            if 'image' in ret and ret['image'] is not None:
                image = ret['image']
                h, w = image.shape[:2]
                y1 = t
                y2 = h - d
                x1 = l
                x2 = w - r
                image = image[y1:y2, x1: x2]
                ret['image'] = image

            if 'bboxes' in ret and ret['bboxes'] is not None:
                bboxes = ret['bboxes']
                bboxes = np.array(bboxes)
                shift = np.array([l, t, l, t])
                bboxes -= shift
                ret['bboxes'] = bboxes

        return ret


class Crop:
    def __init__(self):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__

    def get_add_params(self, dst_coor, w, h):
        x1, x2, y1, y2 = dst_coor
        return {self.name: dict(x1=x1, x2=x2, y1=y1, y2=y2, w=w, h=h)}

    def parse_add_params(self, ret):
        info = ret[self.name]
        x1 = info.get('x1', 0)
        x2 = info.get('x2', 0)
        y1 = info.get('y1', 0)
        y2 = info.get('y2', 0)
        w = info.get('w', 0)
        h = info.get('h', 0)
        return x1, x2, y1, y2, w, h

    def __call__(self, image, dst_coor, bboxes=None, classes=None, **kwargs):
        h, w, c = image.shape
        add_params = self.get_add_params(dst_coor, w, h)
        image = self.apply_image(image, add_params)
        bboxes, classes = self.apply_bboxes_classes(bboxes, classes, add_params)

        return {
            'image': image,
            'bboxes': bboxes,
            'classes': classes,
            **add_params
        }

    def apply_image(self, image, ret):
        x1, x2, y1, y2, w, h = self.parse_add_params(ret)
        return image[y1:y2, x1: x2]

    def apply_bboxes_classes(self, bboxes, classes, ret):
        if bboxes is not None:
            x1, x2, y1, y2, w, h = self.parse_add_params(ret)
            bboxes = bboxes
            shift = np.array([x1, y1, x1, y1])
            bboxes -= shift
            bboxes = bboxes.clip(min=0)
            bboxes[:, 0::2] = np.where(bboxes[:, 0::2] > x2 - x1, x2 - x1, bboxes[:, 0::2])
            bboxes[:, 1::2] = np.where(bboxes[:, 1::2] > y2 - y1, y2 - y1, bboxes[:, 1::2])

            idx = ~((bboxes[:, 0] == bboxes[:, 2]) | (bboxes[:, 1] == bboxes[:, 3]))
            bboxes = bboxes[idx]

            if classes is not None:
                classes = np.array(classes)
                classes = classes[idx]

        return bboxes, classes

    def restore(self, ret):
        x1, x2, y1, y2, w, h = self.parse_add_params(ret)

        if 'image' in ret and ret['image'] is not None:
            # irreversible restore
            pad = Pad()
            image = ret['image']
            image = pad.apply_image(image, {pad.name: dict(t=y1, d=h - y2, l=x1, r=w - x2)})
            ret['image'] = image

        if 'bboxes' in ret and ret['bboxes'] is not None:
            bboxes = ret['bboxes']
            shift = np.array([x1, y1, x1, y1])
            bboxes += shift
            ret['bboxes'] = bboxes

        return ret


class PadCrop:
    def __init__(self, is_pad=True, **pad_kwargs):
        self.is_pad = is_pad
        self.pad = Pad(**pad_kwargs)
        self.crop = Crop()

    def __call__(self, image, dst_coor, bboxes=None, classes=None, **kwargs):
        x1, x2, y1, y2 = dst_coor
        h, w, c = image.shape

        assert x1 >= 0 and y1 >= 0, ValueError(f'{x1 = } and {y1 = } must not be smaller than 0')

        ret = dict(image=image, bboxes=bboxes, classes=classes)

        if x2 > w or y2 > h:
            if self.is_pad:
                ret.update(self.pad(dst=(x2, y2), **ret))

            else:
                raise ValueError(f'image width = {w} and height = {h} must be greater than {x2 = } and {y2 = } or set pad=True')

        ret.update(self.crop(**ret, dst_coor=dst_coor))
        return ret

    def apply_image(self, image, ret):
        image = self.pad.apply_image(image, ret)
        return self.crop.apply_image(image, ret)

    def apply_bboxes_classes(self, bboxes, classes, ret):
        return self.apply_bboxes_classes(bboxes, classes, ret)

    def restore(self, ret):
        ret = self.crop.restore(ret)
        return self.pad.restore(ret)


class Random(PadCrop):
    """See Also `torchvision.transforms.RandomCrop` or `albumentations.RandomCrop`"""

    def __call__(self, image, dst, **kwargs):
        """(w, h) -> (dst, dst)"""
        h, w, c = image.shape

        w_ = np.random.randint(w - dst) if w > dst else 0
        h_ = np.random.randint(h - dst) if h > dst else 0

        return super().__call__(image, (w_, w_ + dst, h_, h_ + dst), **kwargs)


class Corner(PadCrop):
    def __init__(self, pos=(LEFT, TOP), is_pad=True, **pad_kwargs):
        self.pos = pos
        super().__init__(is_pad=is_pad, **pad_kwargs)

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

        return super().__call__(image, (w_, w_ + dst, h_, h_ + dst), **kwargs)


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
        rets = self.apply(image=image, dst=dst, bboxes=bboxes)
        _ = self.shift(image, bboxes)
        image, bboxes = _['image'], _['bboxes']

        _rets = self.apply(image=image, dst=dst, bboxes=bboxes, classes=classes)
        rets += _rets

        return rets


class Center(PadCrop):
    """See Also `torchvision.transforms.CenterCrop` or `albumentations.CenterCrop`"""

    def __call__(self, image, dst, **kwargs):
        h, w, c = image.shape

        w_ = max(w - dst, 0) // 2
        h_ = max(h - dst, 0) // 2

        return super().__call__(image, (w_, w_ + dst, h_, h_ + dst), **kwargs)
