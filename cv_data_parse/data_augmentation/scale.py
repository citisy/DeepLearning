"""change the shape of image by resizing the image according to some algorithm"""
import cv2
import numpy as np
from . import crop

interpolation_mode = [
    cv2.INTER_LINEAR,
    cv2.INTER_NEAREST,
    cv2.INTER_AREA,
    cv2.INTER_CUBIC,
    cv2.INTER_LANCZOS4
]


def proportion(image, dst, interpolation=0):
    """proportional scale the shortest edge to destination size
    See Also `torchvision.transforms.Resize`"""
    h, w, c = image.shape
    a = min(w, h)
    p = dst / a

    return cv2.resize(image, None, fx=p, fy=p, interpolation=interpolation_mode[interpolation])


def rectangle(image, dst, interpolation=0):
    """scale to special dst * dst
    See Also `torchvision.transforms.Resize`"""
    return cv2.resize(image, (dst, dst), interpolation=interpolation_mode[interpolation])


def jitter(image, dst, size_range=(256, 384)):
    """See Also `torchvision.transforms.RandomResizedCrop`"""
    s = np.random.randint(*size_range)
    image = proportion(image, s)
    image = crop.random(image, dst, is_pad=True)

    return image
