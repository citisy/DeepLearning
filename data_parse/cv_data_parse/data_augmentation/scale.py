"""change the shape of image by resizing the image according to some algorithm"""
import cv2
import numpy as np
from . import crop, Apply

interpolation_mode = [
    cv2.INTER_LINEAR,
    cv2.INTER_NEAREST,
    cv2.INTER_AREA,
    cv2.INTER_CUBIC,
    cv2.INTER_LANCZOS4
]

SHORTEST, LONGEST = 1, 2
AUTO = 3


class Proportion:
    """proportional scale the choice edge to destination size
    See Also `torchvision.transforms.Resize` or `albumentations.Resize`"""

    def __init__(self, interpolation=0, choice_edge=SHORTEST, max_ratio=None):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.interpolation = interpolation_mode[interpolation]
        self.choice_edge = choice_edge
        self.max_ratio = max_ratio

    def get_params(self, dst, w, h):
        if self.choice_edge == SHORTEST:
            p = dst / min(w, h)
        elif self.choice_edge == LONGEST:
            p = dst / max(w, h)
        elif self.choice_edge == AUTO:  # choice the min scale factor
            p1 = abs(dst - w) / w
            p2 = abs(dst - h) / h
            p = dst / w if p1 < p2 else dst / h
        else:
            raise ValueError(f'dont support {self.choice_edge = }')

        if self.max_ratio:
            # set in [1 / (1 + self.max_ratio), 1 + self.max_ratio]
            p = max(min(p, 1 + self.max_ratio), 1 / (1 + self.max_ratio))
        return p

    def get_add_params(self, dst, w, h):
        p = self.get_params(dst, w, h)
        return {self.name: dict(p=p)}

    def parse_add_params(self, ret):
        return ret[self.name]['p']

    def __call__(self, image, dst, bboxes=None, **kwargs):
        h, w, c = image.shape
        add_params = self.get_add_params(dst, w, h)

        return {
            'image': self.apply_image(image, add_params),
            'bboxes': self.apply_bboxes(image, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        p = self.parse_add_params(ret)
        return cv2.resize(image, None, fx=p, fy=p, interpolation=self.interpolation)

    def apply_bboxes(self, bboxes, ret):
        p = self.parse_add_params(ret)
        if bboxes is not None:
            bboxes = np.array(bboxes, dtype=float) * p
            bboxes = bboxes.astype(int)
        return bboxes

    def restore(self, ret):
        p = self.parse_add_params(ret)
        if 'image' in ret and ret['image'] is not None:
            image = ret['image']
            image = cv2.resize(image, None, fx=1 / p, fy=1 / p, interpolation=self.interpolation)
            ret['image'] = image

        if 'bboxes' in ret and ret['bboxes'] is not None:
            bboxes = ret['bboxes']
            bboxes = np.array(bboxes, dtype=float) / p
            bboxes = bboxes.astype(int)
            ret['bboxes'] = bboxes

        return ret


class Rectangle:
    """scale h * w to dst * dst
    See Also `torchvision.transforms.Resize` or `albumentations.Resize`"""

    def __init__(self, interpolation=0):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.interpolation = interpolation_mode[interpolation]

    def get_params(self, dst, w, h):
        return dst / w, dst / h

    def get_add_params(self, dst, w, h):
        pw, ph = self.get_params(dst, w, h)
        return {self.name: dict(pw=pw, ph=ph)}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['pw'], info['ph']

    def __call__(self, image, dst, bboxes=None, **kwargs):
        h, w, c = image.shape
        add_params = self.get_add_params(dst, w, h)

        return {
            'image': self.apply_image(image, add_params),
            'bboxes': self.apply_bboxes(image, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        pw, ph = self.parse_add_params(ret)
        return cv2.resize(image, None, fx=pw, fy=ph, interpolation=self.interpolation)

    def apply_bboxes(self, bboxes, ret):
        if bboxes is not None:
            pw, ph = self.parse_add_params(ret)
            bboxes = np.array(bboxes, dtype=float) * np.array([pw, ph, pw, ph])
            bboxes = bboxes.astype(int)

        return bboxes

    def restore(self, ret):
        pw, ph = self.parse_add_params(ret)
        if 'image' in ret and ret['image'] is not None:
            image = ret['image']
            image = cv2.resize(image, None, fx=1 / pw, fy=1 / ph, interpolation=self.interpolation)
            ret['image'] = image

        if 'bboxes' in ret and ret['bboxes'] is not None:
            bboxes = ret['bboxes']
            bboxes = np.array(bboxes, dtype=float) / np.array([pw, ph, pw, ph])
            bboxes = bboxes.astype(int)
            ret['bboxes'] = bboxes

        return ret


class UnrestrictedRectangle(Rectangle):
    """scale h * w to dst_h * dst_w
    See Also `torchvision.transforms.Resize` or `albumentations.Resize`"""

    def get_params(self, dst_w, dst_h, w, h):
        return dst_w / w, dst_h / h

    def get_add_params(self, dst_w, dst_h, w, h):
        pw, ph = self.get_params(dst_w, dst_h, w, h)
        return {'scale.UnrestrictedRectangle': dict(pw=pw, ph=ph)}


class LetterBox:
    """resize, crop, and pad"""

    def __init__(self, interpolation=0, max_ratio=None, **pad_kwargs):
        self.resize = Proportion(choice_edge=LONGEST, interpolation=interpolation, max_ratio=max_ratio)  # scale to longest edge
        self.crop = crop.Center(is_pad=True, pad_type=2, **pad_kwargs)

    def __call__(self, image, dst, bboxes=None, **kwargs):
        ret = dict(image=image, bboxes=bboxes, dst=dst, **kwargs)
        ret.update(self.resize(**ret))
        ret.update(self.crop(**ret))

        return ret

    def apply_image(self, image, ret):
        image = self.resize.apply_image(image, ret)
        image = self.crop.apply_image(image, ret)
        return image

    def restore(self, ret):
        ret = self.crop.restore(ret)
        ret = self.resize.restore(ret)

        return ret


class Jitter:
    """random resize, crop, and pad
    See Also `torchvision.transforms.RandomResizedCrop`"""

    def __init__(self, size_range=None, interpolation=0, **pad_kwargs):
        """

        Args:
            size_range (None or tuple):
            interpolation:
            **pad_kwargs:
        """
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.size_range = size_range
        self.resize = Proportion(choice_edge=2, interpolation=interpolation)
        self.crop = crop.Random(is_pad=True, pad_type=2, **pad_kwargs)

    def get_add_params(self, hidden_dst):
        return {self.name: dict(dst=hidden_dst)}

    def parse_add_params(self, ret):
        return ret[self.name]['dst']

    def get_params(self, dst):
        size_range = self.size_range if self.size_range else (int(dst * 1.14), int(dst * 1.71))
        return np.random.randint(*size_range)

    def __call__(self, image, dst, bboxes=None, **kwargs):
        hidden_dst = self.get_params(dst)
        ret = self.get_add_params(hidden_dst)
        ret.update(self.resize(image, hidden_dst, bboxes=bboxes))
        ret.update(self.crop(ret['image'], dst, bboxes=ret['bboxes']))

        return ret

    def apply_image(self, image, ret):
        hidden_dst = self.parse_add_params(ret)
        dst = ret['dst']
        ret['dst'] = hidden_dst
        image = self.resize.apply_image(image, ret)
        ret['dst'] = dst
        image = self.crop.apply_image(image, ret)
        return image

    def restore(self, ret):
        ret = self.crop.restore(ret)
        ret = self.resize.restore(ret)

        return ret


class BatchLetterBox:
    def __init__(self, interpolation=0, choice_edge=SHORTEST):
        self.interpolation = interpolation_mode[interpolation]
        self.choice_edge = choice_edge
        self.aug = LetterBox(interpolation=interpolation)

    def __call__(self, image_list, bboxes_list=None, classes_list=None, **kwargs):
        image_sizes = [image[:2] for image in image_list]

        if self.choice_edge == SHORTEST:
            dst = np.min(image_sizes)
        elif self.choice_edge == LONGEST:
            dst = np.max(image_sizes)
        elif self.choice_edge == AUTO:
            dst = np.mean(image_sizes)
        else:
            raise ValueError(f'dont support {self.choice_edge = }')

        rets = []

        for i in range(len(image_list)):
            ret = dict(
                image=image_list[i],
                bboxes=bboxes_list[i],
                classes=classes_list[i],
                dst=dst
            )

            for k, v in kwargs.items():
                ret[k] = v

            ret.update(self.aug(**ret))
            rets.append(ret)

        return rets
