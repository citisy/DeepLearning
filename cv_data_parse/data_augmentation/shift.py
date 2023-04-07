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


class HFlip:
    """See Also `torchvision.transforms.RandomHorizontalFlip`"""

    def __call__(self, image):
        return cv2.flip(image, 1)


class VFlip:
    """See Also `torchvision.transforms.RandomVerticalFlip`"""

    def __call__(self, image):
        return cv2.flip(image, 0)


class RandomHShift:
    def __init__(self, ignore_overlap=True, shift_class=None):
        self.ignore_overlap = ignore_overlap
        self.shift_class = shift_class

    def __call__(self, image, bboxes, classes):
        # check and select bbox
        new_bboxes = np.array(bboxes, dtype=int)

        flag = self.check_coor_overlap(new_bboxes[:, 1], new_bboxes[:, 3])

        if isinstance(self.shift_class, int):
            shift_flag = classes == self.shift_class
        elif self.shift_class is None:
            shift_flag = np.ones(len(classes), dtype=bool)
        else:
            shift_flag = np.zeros(len(classes), dtype=bool)
            for c in self.shift_class:
                shift_flag |= classes == c

        if np.any(shift_flag & flag):
            if self.ignore_overlap:
                shift_flag = shift_flag & (~flag)
            else:
                raise ValueError('shift area must be not overlapped')

        img = np.zeros_like(image, dtype=image.dtype)

        _bboxes = new_bboxes[shift_flag]
        shift_bbox = _bboxes.copy()
        non_bbox = new_bboxes[~shift_flag].copy()
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
        new_bboxes = np.concatenate([non_bbox, shift_bbox], axis=0)

        if classes is not None:
            classes = np.concatenate([classes[~shift_flag], classes[shift_flag]])

        return img, new_bboxes.astype(bboxes.dtype), classes

    @staticmethod
    def check_coor_overlap(a, b):
        f1 = (a[:, None] < a[None, :]) & (b[:, None] > a[None, :])
        f2 = (b[:, None] > a[None, :]) & (b[:, None] < b[None, :])

        return np.any(f1 | f2, axis=1)


class RandomVShift(RandomHShift):
    def __call__(self, image, bboxes, classes):
        bboxes = bboxes.copy()
        image = image.copy()
        image = image.T
        bboxes[:, (0, 1, 2, 3)] = bboxes[:, (1, 0, 3, 2)]

        image, bboxes, classes = super().__call__(image, bboxes, classes)

        image = image.T
        bboxes[:, (0, 1, 2, 3)] = bboxes[:, (1, 0, 3, 2)]

        return image, bboxes, classes


class Rotate:
    """Rotates the image by angle.
    See Also `torchvision.transforms.RandomRotation`

    Args:
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

    def __init__(self, angle=90, interpolation=0, expand=False, center=None, fill=0):
        self.angle = angle
        self.interpolation = interpolation_mode[interpolation]
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def transform(x, y, matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    def __call__(self, image):
        h, w, c = image.shape
        center = self.center or (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, self.angle, 1)

        if self.expand:
            # calculate output size
            xx = []
            yy = []

            angle = -np.radians(self.angle)
            expand_matrix = [
                round(np.cos(angle), 15),
                round(np.sin(angle), 15),
                0.0,
                round(-np.sin(angle), 15),
                round(np.cos(angle), 15),
                0.0,
            ]

            post_trans = (0, 0)
            expand_matrix[2], expand_matrix[5] = self.transform(
                -center[0] - post_trans[0], -center[1] - post_trans[1],
                expand_matrix)
            expand_matrix[2] += center[0]
            expand_matrix[5] += center[1]

            for x, y in ((0, 0), (w, 0), (w, h), (0, h)):
                x, y = self.transform(x, y, expand_matrix)
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
            flags=self.interpolation,
            borderValue=self.fill
        )
