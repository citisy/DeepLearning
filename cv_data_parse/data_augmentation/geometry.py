"""flip, move, rotate the image without changing the shape of image, etc."""
import cv2
import numpy as np
import numbers
from utils import visualize

interpolation_mode = [
    cv2.INTER_LINEAR,
    cv2.INTER_NEAREST,
    cv2.INTER_AREA,
    cv2.INTER_CUBIC,
    cv2.INTER_LANCZOS4
]

fill_mode = [
    cv2.BORDER_CONSTANT,
    cv2.BORDER_REPLICATE,
    cv2.BORDER_REFLECT_101,
    cv2.BORDER_REFLECT,
    cv2.BORDER_WRAP
]


class HFlip:
    """See Also `torchvision.transforms.RandomHorizontalFlip`"""

    def __call__(self, image, bboxes=None, **kwargs):
        h, w = image.shape[:2]
        image = cv2.flip(image, 1)

        if bboxes is not None:
            bboxes = np.array(bboxes)
            bboxes[:, 0::2] = w - bboxes[:, 0::2]

        return dict(
            image=image,
            bboxes=bboxes
        )


class VFlip:
    """See Also `torchvision.transforms.RandomVerticalFlip`"""

    def __call__(self, image, bboxes=None, **kwargs):
        h, w = image.shape[:2]
        image = cv2.flip(image, 0)

        if bboxes is not None:
            bboxes = np.array(bboxes)
            bboxes[:, 1::2] = h - bboxes[:, 1::2]

        return dict(
            image=image,
            bboxes=bboxes
        )


class RandomVShift:
    def __init__(self, ignore_overlap=True, shift_class=None):
        self.ignore_overlap = ignore_overlap
        self.shift_class = shift_class

    def __call__(self, image, bboxes, classes=None, **kwargs):
        # check and select bbox
        new_bboxes = np.array(bboxes, dtype=int)
        classes = classes.copy()

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
                raise ValueError('shift area must be not overlapped, or set `ignore_overlap=True` while init the class')

        shift_img = np.zeros_like(image, dtype=image.dtype)

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
            shift_img[new_start: new_end] = image[old_start: a]

            delta += (d - c) - (b - a)
            new_b = b + delta

            shift_img[new_end: new_b] = image[c: d]

            shift_bbox[idx[i], (1, 3)] = (new_end, new_b)
            non_delta[(non_bbox[:, 1] + non_delta[:, 0] > new_end)] = delta

            new_start = new_b
            old_start = b

        shift_img[new_start:] = image[old_start:]
        non_bbox[:, (1, 3)] += non_delta.astype(int)
        new_bboxes = np.concatenate([non_bbox, shift_bbox], axis=0)

        if classes is not None:
            classes = np.concatenate([classes[~shift_flag], classes[shift_flag]])

        return dict(
            image=shift_img,
            bboxes=new_bboxes.astype(bboxes.dtype),
            classes=classes
        )

    @staticmethod
    def check_coor_overlap(a, b):
        """box1 = (ya1, yb1), box2 = (ya2, yb2),
        which were overlap in y axis where
        (ya1 < ya2 & yb1 > ya2) | (yb1 > ya2 & yb1 < yb2)
        """
        f1 = (a[:, None] <= a[None, :]) & (b[:, None] >= a[None, :])
        f2 = (b[:, None] >= a[None, :]) & (b[:, None] <= b[None, :])
        _ = list(range(len(a)))
        f1[_, _] = False
        f2[_, _] = False

        return np.any(f1 | f2, axis=1) | np.any(f1 | f2, axis=0)


class RandomHShift(RandomVShift):
    def __call__(self, image, bboxes, classes=None, **kwargs):
        bboxes = bboxes.copy()
        image = image.copy()
        image = image.T
        bboxes[:, (0, 1, 2, 3)] = bboxes[:, (1, 0, 3, 2)]

        ret = super().__call__(image, bboxes, classes)

        ret['image'] = ret['image'].T
        ret['bboxes'][:, (0, 1, 2, 3)] = ret['bboxes'][:, (1, 0, 3, 2)]

        return ret


class Rotate:
    """Rotates the image by angle.

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

    def get_params(self):
        return self.angle

    @staticmethod
    def transform(x, y, matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    def __call__(self, image, bboxes=None, **kwargs):
        angle = self.get_params()

        h, w, c = image.shape

        center = self.center or (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1)

        if self.expand:
            # calculate output size
            xx = []
            yy = []

            _angle = -np.radians(angle)
            expand_matrix = [
                round(np.cos(_angle), 15),
                round(np.sin(_angle), 15),
                0.0,
                round(-np.sin(_angle), 15),
                round(np.cos(_angle), 15),
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

        image = cv2.warpAffine(
            image,
            M, (w, h),
            flags=self.interpolation,
            borderValue=self.fill
        )

        if bboxes is not None:
            n = len(bboxes)
            xy = np.ones((n * 4, 3))
            xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, w)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, h)

            bboxes = new

        return {
            'image': image,
            'bboxes': bboxes,
            'geo.Rotate': dict(
                angle=angle,
            )}


class RandomRotate(Rotate):
    """See Also `torchvision.transforms.RandomRotation`"""

    def get_params(self):
        if isinstance(self.angle, numbers.Number):
            angle = int(np.random.uniform(-self.angle, self.angle))
        else:
            angle = int(np.random.uniform(self.angle[0], self.angle[1]))

        return angle


class Affine:
    """rotate, scale and shift the image"""
    def __init__(self, angle=45, translate=(0., 0.), interpolation=0, fill_type=0, scale=1., shear=0.):
        self.angle = angle
        self.translate = translate
        self.interpolation = interpolation_mode[interpolation]
        self.fill_type = fill_mode[fill_type]
        self.scale = scale
        self.shear = shear

    def get_params(self):
        return self.angle, self.translate, self.scale, self.shear

    def _get_inverse_affine_matrix(self, center, angle, translate, scale, shear):
        # https://github.com/pytorch/vision/blob/v0.4.0/torchvision/transforms/functional.py#L717
        if isinstance(shear, numbers.Number):
            shear = [shear, 0]

        rot = np.radians(angle)
        sx, sy = [np.radians(s) for s in shear]

        cx, cy = center
        tx, ty = translate

        # RSS without scaling
        a = np.cos(rot - sy) / np.cos(sy)
        b = -np.cos(rot - sy) * np.tan(sx) / np.cos(sy) - np.sin(rot)
        c = np.sin(rot - sy) / np.cos(sy)
        d = -np.sin(rot - sy) * np.tan(sx) / np.cos(sy) + np.cos(rot)

        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        M = [d, -b, 0, -c, a, 0]
        M = [x / scale for x in M]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        M[2] += M[0] * (-cx - tx) + M[1] * (-cy - ty)
        M[5] += M[3] * (-cx - tx) + M[4] * (-cy - ty)

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        M[2] += cx
        M[5] += cy
        return M

    def __call__(self, image, bboxes=None, **kwargs):
        h, w = image.shape[:2]
        angle, translate, scale, shear = self.get_params()

        M = self._get_inverse_affine_matrix((w / 2, h / 2), angle, (0, 0), scale, shear)
        M = np.array(M).reshape(2, 3)

        startpoints = [(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)]
        project = lambda x, y, a, b, c: int(a * x + b * y + c)
        endpoints = [(project(x, y, *M[0]), project(x, y, *M[1])) for x, y in startpoints]

        rect = cv2.minAreaRect(np.array(endpoints))
        bbox = cv2.boxPoints(rect).astype(int)
        max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
        min_x, min_y = bbox[:, 0].min(), bbox[:, 1].min()

        dst_w = int(max_x - min_x)
        dst_h = int(max_y - min_y)
        M[0, 2] += (dst_w - w) / 2
        M[1, 2] += (dst_h - h) / 2

        # add translate
        dst_w += int(abs(translate[0]))
        dst_h += int(abs(translate[1]))
        if translate[0] < 0:
            M[0, 2] += abs(translate[0])
        if translate[1] < 0:
            M[1, 2] += abs(translate[1])

        image = cv2.warpAffine(
            image,
            M, (dst_w, dst_h),
            flags=self.interpolation,
            borderMode=self.fill_type,
            borderValue=(114, 114, 114)
        )

        if bboxes is not None:
            n = len(bboxes)
            xy = np.ones((n * 4, 3))
            xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, dst_w)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, dst_h)

            bboxes = new

        return {
            'image': image,
            'bboxes': bboxes,
            'geo.Affine': dict(
                angle=angle,
                translate=translate,
                scale=scale,
                shear=shear
            )}


class RandomAffine(Affine):
    """see also `torchvision.transforms.RandomAffine`"""

    def get_params(self):
        if isinstance(self.angle, numbers.Number):
            angle = int(np.random.uniform(-self.angle, self.angle))
        else:
            angle = int(np.random.uniform(self.angle[0], self.angle[1]))

        if isinstance(self.translate, numbers.Number):
            translate = np.array([-self.translate, self.translate])
        else:
            translate = np.array(self.translate)

        translate = np.random.uniform(*translate, size=2).astype(int)

        return angle, translate, self.scale, self.shear


class Perspective:
    def __init__(self, distortion=0.25, interpolation=0, fill_type=0, keep_shape=True):
        self.distortion = distortion
        self.interpolation = interpolation_mode[interpolation]
        self.fill_type = fill_mode[fill_type]
        self.keep_shape = keep_shape

    def get_params(self, h, w):
        offset_h = self.distortion * h
        offset_w = self.distortion * w

        return np.array([
            (0, 0),
            (w - 1, 0),
            (w - 1 - offset_w, h - 1 - offset_h),
            (offset_w, h - 1 - offset_h)
        ], dtype=np.float32)

    def __call__(self, image, bboxes=None, **kwargs):
        h, w = image.shape[:2]
        end_points = self.get_params(h, w)
        start_points = np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], dtype=np.float32)

        M = cv2.getPerspectiveTransform(start_points, end_points)

        image = cv2.warpPerspective(
            image,
            M, (w, h),
            flags=self.interpolation,
            borderMode=self.fill_type,
            borderValue=(114, 114, 114)
        )

        if not self.keep_shape:
            rect = cv2.minAreaRect(end_points)
            bbox = cv2.boxPoints(rect).astype(dtype=np.int)
            max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
            min_x, min_y = bbox[:, 0].min(), bbox[:, 1].min()
            min_x, min_y = max(min_x, 0), max(min_y, 0)
            image = image[min_y:max_y, min_x:max_x]

        if bboxes is not None:
            n = len(bboxes)
            xy = np.ones((n * 4, 3))
            xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, w)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, h)

            bboxes = new

        return {
            'image': image,
            'bboxes': bboxes,
            'geo.Perspective': dict(
                end_points=end_points
            )}


class RandomPerspective(Perspective):
    """see also `torchvision.transforms.RandomPerspective`"""

    def get_params(self, h, w):
        if isinstance(self.distortion, numbers.Number):
            distortion = np.array([-self.distortion, self.distortion])
        else:
            distortion = np.array(self.distortion)

        _offset_h = distortion * h
        _offset_w = distortion * w

        offset_h = np.random.uniform(*_offset_h, size=4)
        offset_w = np.random.uniform(*_offset_w, size=4)

        return np.array([
            (offset_w[0], offset_h[0]),
            (w - 1 - offset_w[1], offset_h[1]),
            (w - 1 - offset_w[2], h - 1 - offset_h[2]),
            (offset_w[3], h - 1 - offset_h[3])
        ], dtype=np.float32)
