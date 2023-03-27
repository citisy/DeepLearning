"""flip, move, rotate the image without changing the shape of image, etc."""
import cv2
import numpy as np

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


def random_h_shift(image, coor):
    img = np.zeros_like(image, dtype=image.dtype)

    sort_coor = sorted(coor, key=lambda x: x[0])

    idx = np.random.choice(range(len(coor)), len(coor), replace=False)

    start_old, start_new = 0, 0
    for i in range(len(coor)):
        a, b = sort_coor[i]
        c, d = coor[idx[i]]

        end_new = start_new + a - start_old
        img[start_new: end_new] = image[start_old: a]

        start_new = end_new
        end_new = start_new + d - c
        img[start_new: end_new] = image[c: d]

        start_new = end_new
        start_old = b

    img[start_new:] = image[start_old:]

    return img


def random_v_shift(image, coor):
    img = np.zeros_like(image, dtype=image.dtype)

    sort_coor = sorted(coor, key=lambda x: x[0])

    idx = np.random.choice(range(len(coor)), len(coor), replace=False)

    start_old, start_new = 0, 0
    for i in range(len(coor)):
        a, b = sort_coor[i]
        c, d = coor[idx[i]]

        end_new = start_new + a - start_old
        img[:, start_new: end_new] = image[:, start_old: a]

        start_new = end_new
        end_new = start_new + d - c
        img[:, start_new: end_new] = image[:, c: d]

        start_new = end_new
        start_old = b

    img[:, start_new:] = image[:, start_old:]

    return img


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
