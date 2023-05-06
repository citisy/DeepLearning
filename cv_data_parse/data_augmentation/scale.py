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


class Proportion:
    """proportional scale the shortest edge to destination size
    See Also `torchvision.transforms.Resize` or `albumentations.Resize`"""

    def __init__(self, interpolation=0):
        self.interpolation = interpolation_mode[interpolation]

    def __call__(self, image, dst, bboxes=None, **kwargs):
        h, w, c = image.shape
        a = min(w, h)
        p = dst / a
        image = cv2.resize(image, None, fx=p, fy=p, interpolation=self.interpolation)

        if bboxes is not None:
            bboxes = np.array(bboxes)
            bboxes *= p

        return {
            'image': image,
            'bboxes': bboxes,
            'scale.Proportion': dict(p=p)
        }


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
            bboxes = np.array(bboxes)
            bboxes[:, 0::2] *= pw
            bboxes[:, 1::2] *= ph

        return {
            'image': image,
            'bboxes': bboxes,
            'scale.Rectangle': dict(pw=pw, ph=ph)
        }


class LetterBox:
    """resize, crop, and pad"""

    def __init__(self):
        self.apply = Apply([
            Proportion(),
            crop.Center(is_pad=True, pad_type=2)
        ])

    def __call__(self, image, dst, bboxes=None, **kwargs):
        return self.apply(image=image, dst=dst, bboxes=bboxes)


class Jitter:
    """random resize, crop, and pad
    See Also `torchvision.transforms.RandomResizedCrop`"""

    def __init__(self, size_range=(256, 384)):
        self.size_range = size_range
        self.resize = Proportion()
        self.crop = crop.Random(is_pad=True, pad_type=2)

    def __call__(self, image, dst, bboxes=None, **kwargs):
        s = np.random.randint(*self.size_range)
        ret = {'scale.Jitter': dict(dst=s)}
        ret.update(self.resize(image, s, bboxes=bboxes))
        ret.update(self.crop(ret['image'], dst, bboxes=ret['bboxes']))

        return ret
