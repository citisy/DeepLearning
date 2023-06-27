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

    def __init__(self, interpolation=0, choice_edge=SHORTEST):
        self.interpolation = interpolation_mode[interpolation]
        self.choice_edge = choice_edge

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

        return p

    def __call__(self, image, dst, bboxes=None, **kwargs):
        h, w, c = image.shape
        p = self.get_params(dst, w, h)
        image = cv2.resize(image, None, fx=p, fy=p, interpolation=self.interpolation)

        if bboxes is not None:
            bboxes = np.array(bboxes, dtype=float) * p
            bboxes = bboxes.astype(int)

        return {
            'image': image,
            'bboxes': bboxes,
            'scale.Proportion': dict(p=p)
        }

    @staticmethod
    def restore(ret):
        params = ret.get('scale.Proportion')
        bboxes = ret['bboxes']
        p = params['p']
        bboxes = np.array(bboxes, dtype=float) / p
        bboxes = bboxes.astype(int)
        ret['bboxes'] = bboxes

        return ret


class Rectangle:
    """scale to special dst * dst
    See Also `torchvision.transforms.Resize` or `albumentations.Resize`"""

    def __init__(self, interpolation=0):
        self.interpolation = interpolation_mode[interpolation]

    def __call__(self, image, dst, bboxes=None, **kwargs):
        h, w, c = image.shape

        image = cv2.resize(image, (dst, dst), interpolation=self.interpolation)

        pw = dst / w
        ph = dst / h

        if bboxes is not None:
            bboxes = np.array(bboxes, dtype=float) * np.array([pw, ph, pw, ph])
            bboxes = bboxes.astype(int)

        return {
            'image': image,
            'bboxes': bboxes,
            'scale.Rectangle': dict(pw=pw, ph=ph)
        }

    @staticmethod
    def restore(ret):
        params = ret.get('scale.Rectangle')
        bboxes = ret['bboxes']
        pw, ph = params['pw'], params['ph']
        bboxes = np.array(bboxes, dtype=float) / np.array([pw, ph, pw, ph])
        bboxes = bboxes.astype(int)
        ret['bboxes'] = bboxes

        return ret


class LetterBox:
    """resize, crop, and pad"""

    def __init__(self, interpolation=0, ):
        self.resize = Proportion(choice_edge=2, interpolation=interpolation)  # scale to longest edge
        self.crop = crop.Center(is_pad=True, pad_type=2)

    def __call__(self, image, dst, bboxes=None, **kwargs):
        ret = dict(image=image, bboxes=bboxes, dst=dst, **kwargs)
        ret.update(self.resize(**ret))
        ret.update(self.crop(**ret))

        return ret

    def restore(self, ret):
        ret = self.crop.restore(ret)
        ret = self.resize.restore(ret)

        return ret


class Jitter:
    """random resize, crop, and pad
    See Also `torchvision.transforms.RandomResizedCrop`"""

    def __init__(self, size_range=(256, 384)):
        self.size_range = size_range
        self.resize = Proportion()
        self.crop = crop.Random(is_pad=True, pad_type=2)

    def get_params(self, dst):
        return self.size_range if self.size_range else (int(dst * 1.14), int(dst * 1.71))

    def __call__(self, image, dst, bboxes=None, **kwargs):
        size_range = self.get_params(dst)
        s = np.random.randint(size_range)
        ret = {'scale.Jitter': dict(dst=s)}
        ret.update(self.resize(image, s, bboxes=bboxes))
        ret.update(self.crop(ret['image'], dst, bboxes=ret['bboxes']))

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

