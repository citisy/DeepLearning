"""flip, move, rotate the image without changing the shape of image, etc."""
import cv2
import numpy as np
from metrics.object_detection import Iou
from utils.visualize import ImageVisualize, RECTANGLE

interpolation_mode = [
    cv2.INTER_LINEAR,
    cv2.INTER_NEAREST,
    cv2.INTER_AREA,
    cv2.INTER_CUBIC,
    cv2.INTER_LANCZOS4
]


def hflip(image):
    """See Also `torchvision.transforms.RandomHorizontalFlip`"""
    return cv2.flip(image, 1)


def vflip(image):
    """See Also `torchvision.transforms.RandomVerticalFlip`"""
    return cv2.flip(image, 0)


def check_coor_overlap(a, b):
    f1 = (a[:, None] < a[None, :]) & (b[:, None] > a[None, :])
    f2 = (b[:, None] > a[None, :]) & (b[:, None] < b[None, :])

    return np.any(f1 | f2, axis=1)


def random_h_shift(image, bboxes, ignore_overlap=True, shift_class=None, classes=None):
    # check and select bbox
    bboxes = np.array(bboxes)

    flag = check_coor_overlap(bboxes[:, 1], bboxes[:, 3])

    if isinstance(shift_class, int):
        shift_flag = classes == shift_class
    elif shift_class is None:
        shift_flag = np.ones(len(classes), dtype=bool)
    else:
        shift_flag = np.zeros(len(classes), dtype=bool)
        for c in shift_class:
            shift_flag |= classes == c

    if np.any(shift_flag & flag):
        if ignore_overlap:
            shift_flag = shift_flag & (~flag)
        else:
            raise ValueError('shift area must be not overlapped')

    img = np.zeros_like(image, dtype=image.dtype)

    _bboxes = bboxes[shift_flag]
    shift_bbox = _bboxes.copy()
    non_bbox = bboxes[~shift_flag].copy()
    non_delta = np.zeros((len(non_bbox), 2))

    argidx = np.argsort(_bboxes[:, 1])

    idx = np.random.choice(range(len(argidx)), len(argidx), replace=False)

    new_start, new_end, old_start, old_end = 0, 0, 0, 0
    delta = 0

    for i in range(len(idx)):
        a, b = _bboxes[argidx[i], (1, 3)]
        c, d = _bboxes[idx[i], (1, 3)]

        new_end = a + delta
        img[new_start: new_end] = image[old_start: a]

        delta += (d - c) - (b - a)
        new_b = b + delta

        img[new_end: new_b] = image[c: d]

        shift_bbox[idx[i], (1, 3)] = (new_end, new_b)
        non_delta[(non_bbox[:, 1] > new_b)] = delta

        if classes is not None:
            classes[idx[i]] = classes[argidx[i]]

        new_start = new_b
        old_start = b

    img[new_start:] = image[old_start:]
    non_bbox[:, (1, 3)] += non_delta.astype(int)
    bboxes = np.concatenate([non_bbox, shift_bbox], axis=0)

    if classes is not None:
        classes = np.concatenate([classes[~shift_flag], classes[shift_flag]])

    return img, bboxes, classes


def random_v_shift(image, bboxes, ignore_overlap=True, shift_class=None, classes=None):
    bboxes = bboxes.copy()
    image = image.copy()
    image = image.T
    bboxes[:, (0, 1, 2, 3)] = bboxes[:, (1, 0, 3, 2)]

    image, bboxes, classes = random_h_shift(image, bboxes, ignore_overlap, shift_class, classes)

    image = image.T
    bboxes[:, (0, 1, 2, 3)] = bboxes[:, (1, 0, 3, 2)]

    return image, bboxes, classes


def rotate(image,
           angle=90,
           interpolation=0,
           expand=False,
           center=None,
           fill=0):
    """Rotates the image by angle.
    See Also `torchvision.transforms.RandomRotation`

    Args:
        image (np.array): Image to be rotated.
        angle (float or int): In degrees counter-clockwise order.
        interpolation (int|str, optional): Interpolation method. index of interpolation_mode
        expand (bool, optional): Optional expansion flag.
            If true, expands the output image to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expanding flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.

    Returns:
        np.array: Rotated image.

    """
    h, w, c = image.shape
    center = center or (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1)

    if expand:
        def transform(x, y, matrix):
            (a, b, c, d, e, f) = matrix
            return a * x + b * y + c, d * x + e * y + f

        # calculate output size
        xx = []
        yy = []

        angle = -np.radians(angle)
        expand_matrix = [
            round(np.cos(angle), 15),
            round(np.sin(angle), 15),
            0.0,
            round(-np.sin(angle), 15),
            round(np.cos(angle), 15),
            0.0,
        ]

        post_trans = (0, 0)
        expand_matrix[2], expand_matrix[5] = transform(
            -center[0] - post_trans[0], -center[1] - post_trans[1],
            expand_matrix)
        expand_matrix[2] += center[0]
        expand_matrix[5] += center[1]

        for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
            x, y = transform(x, y, expand_matrix)
            xx.append(x)
            yy.append(y)
        nw = np.ceil(max(xx)) - np.floor(min(xx))
        nh = np.ceil(max(yy)) - np.floor(min(yy))

        M[0, 2] += (nw - w) * 0.5
        M[1, 2] += (nh - h) * 0.5

        w, h = int(nw), int(nh)

    return cv2.warpAffine(
        image,
        M, (w, h),
        flags=interpolation_mode[interpolation],
        borderValue=fill
    )
