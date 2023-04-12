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
    See Also `torchvision.transforms.Resize`"""

    def __init__(self, interpolation=0):
        self.interpolation = interpolation_mode[interpolation]

    def __call__(self, image, dst):
        h, w, c = image.shape
        a = min(w, h)
        p = dst / a
        image = cv2.resize(image, None, fx=p, fy=p, interpolation=self.interpolation)

        return dict(
            image=image,
            p=p
        )


class Rectangle:
    """scale to special dst * dst
    See Also `torchvision.transforms.Resize`"""

    def __init__(self, interpolation=0):
        self.interpolation = interpolation_mode[interpolation]

    def __call__(self, image, dst):
        image = cv2.resize(image, (dst, dst), interpolation=self.interpolation)

        return dict(
            image=image
        )


class LetterBox:
    """resize, crop, and pad"""

    def __init__(self):
        self.apply = Apply([
            Proportion(),
            crop.Center(is_pad=True, pad_type=2)
        ])

    def __call__(self, image, dst):
        return self.apply(image, dst)


class Jitter:
    """random resize, crop, and pad
    See Also `torchvision.transforms.RandomResizedCrop`"""

    def __init__(self, size_range=(256, 384)):
        self.size_range = size_range
        self.resize = Proportion()
        self.crop = crop.Random(is_pad=True, pad_type=2)

    def __call__(self, image, dst):
        s = np.random.randint(*self.size_range)
        ret = dict(_dst=s)
        ret.update(self.resize(image, s))
        ret.update(self.crop(ret['image'], dst))

        return ret
