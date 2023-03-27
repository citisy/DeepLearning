"""change the shape of image by cutting the image directly"""
import cv2
import numpy as np

fill_mode = [
    cv2.BORDER_CONSTANT,
    cv2.BORDER_REPLICATE,
    cv2.BORDER_REFLECT_101,
    cv2.BORDER_REFLECT,
    cv2.BORDER_WRAP
]

LEFT, RIGHT, TOP, DOWN = 0, 1, 0, 1


def crop(image, bbox, is_pad=True, **pad_kwargs):
    x1, x2, y1, y2 = bbox
    h, w, c = image.shape

    assert x1 >= 0 and y1 >= 0, ValueError(f'{x1 = } and {y1 = } must not be smaller than 0')

    if x2 > w or y2 > h:
        if is_pad:
            if x2 > w:
                image = pad(image, x2, direction='w', **pad_kwargs)
            if y2 > h:
                image = pad(image, y2, direction='h', **pad_kwargs)
        else:
            raise ValueError(f'image width = {w} and height = {h} must be greater than {x2 = } and {y2 = } or set pad=True')

    return image[y1:y2, x1: x2]


def pad(image, dst, direction='w', pad_type=1, fill_type=0, fill=0):
    """See Also `torchvision.transforms.Pad`

    Args:
        image: (h, w, c) image
        dst: padding to destination size
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
        fill: fill value

    Returns:

    """
    h, w, c = image.shape

    if direction == 'w':
        if pad_type == 0:
            l, r = dst - w, 0
        elif pad_type == 1:
            l, r = 0, dst - w
        elif pad_type == 2:
            l = (dst - w) // 2
            r = dst - w - l
        elif pad_type == 3:
            l = np.random.randint(dst - w)
            r = dst - w - l
        else:
            raise ValueError(f'dont support {pad_type = }')

        image = cv2.copyMakeBorder(
            image, 0, 0, l, r,
            borderType=fill_mode[fill_type],
            value=fill
        )

    elif direction == 'h':
        if pad_type == 0:
            t, d = dst - h, 0
        elif pad_type == 1:
            t, d = 0, dst - 0
        elif pad_type == 2:
            t = (dst - h) // 2
            d = dst - h - t
        elif pad_type == 3:
            t = np.random.randint(dst - h)
            d = dst - h - t
        else:
            raise ValueError(f'dont support {pad_type = }')

        image = cv2.copyMakeBorder(
            image, t, d, 0, 0,
            borderType=fill_mode[fill_type],
            value=fill
        )

    else:
        raise ValueError(f'dont support {direction = }')

    return image


def random(image, dst, is_pad=True, **pad_kwargs):
    """(w, h) -> (dst, dst)
    See Also `torchvision.transforms.RandomCrop`"""
    h, w, c = image.shape

    w_ = np.random.randint(w - dst) if w > dst else 0
    h_ = np.random.randint(h - dst) if h > dst else 0

    return crop(image, (w_, w_ + dst, h_, h_ + dst), is_pad=is_pad, **pad_kwargs)


def corner(image, dst, pos=(LEFT, TOP), is_pad=True, **pad_kwargs):
    """See Also `torchvision.transforms.FiveCrop` and `torchvision.transforms.TenCrop`"""
    h, w, c = image.shape

    if pos == (LEFT, TOP):
        w_, h_ = 0, 0
    elif pos == (RIGHT, TOP):
        w_, h_ = 0, max(h - dst, 0)
    elif pos == (LEFT, DOWN):
        w_, h_ = max(w - dst, 0), 0
    elif pos == (RIGHT, DOWN):
        w_, h_ = max(w - dst, 0), max(h - dst, 0)
    else:
        raise ValueError(f'dont support {pos = }')

    return crop(image, (w_, w_ + dst, h_, h_ + dst), is_pad=is_pad, **pad_kwargs)


def center(image, dst, is_pad=True, **pad_kwargs):
    """See Also `torchvision.transforms.CenterCrop`"""
    h, w, c = image.shape

    w_ = max(w - dst, 0) // 2
    h_ = max(h - dst, 0) // 2

    return crop(image, (w_, w_ + dst, h_, h_ + dst), is_pad=is_pad, **pad_kwargs)
