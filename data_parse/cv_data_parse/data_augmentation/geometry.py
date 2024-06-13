"""flip, move, rotate the image without changing the shape of image, etc."""
import cv2
import numpy as np
import numbers
from metrics.object_detection import Overlap

LEFT, RIGHT, TOP, DOWN = 0, 1, 0, 1

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
    """See Also `torchvision.transforms.RandomHorizontalFlip` or `albumentations.HorizontalFlip`"""

    def __call__(self, image, bboxes=None, **kwargs):
        h, w = image.shape[:2]
        image = self.apply_image(image)
        bboxes = self.apply_bboxes(bboxes, w)

        return dict(
            image=image,
            bboxes=bboxes
        )

    def apply_image(self, image, *args):
        # note, avoid inplace mode
        image = image.copy()
        return cv2.flip(image, 1, dst=image)

    def apply_bboxes(self, bboxes, w):
        if bboxes is not None:
            bboxes = np.array(bboxes)
            bboxes[:, (0, 2)] = w - bboxes[:, (2, 0)]
        return bboxes

    @staticmethod
    def restore(ret):
        h, w = ret['image'].shape[:2]
        bboxes = ret['bboxes']
        bboxes[:, (0, 2)] = w - bboxes[:, (2, 0)]
        ret['bboxes'] = bboxes

        return ret


class VFlip:
    """See Also `torchvision.transforms.RandomVerticalFlip` or `albumentations.VerticalFlip`"""

    def __call__(self, image, bboxes=None, **kwargs):
        h, w = image.shape[:2]
        image = self.apply_image(image)
        bboxes = self.apply_bboxes(bboxes, h)

        return dict(
            image=image,
            bboxes=bboxes
        )

    def apply_image(self, image, *args):
        # note, avoid inplace mode
        image = image.copy()
        return cv2.flip(image, 0, dst=image)

    def apply_bboxes(self, bboxes, h):
        if bboxes is not None:
            bboxes = np.array(bboxes)
            bboxes[:, (1, 3)] = h - bboxes[:, (3, 1)]

        return bboxes

    @staticmethod
    def restore(ret):
        h, w = ret['image'].shape[:2]
        bboxes = ret['bboxes']
        bboxes[:, (1, 3)] = h - bboxes[:, (3, 1)]
        ret['bboxes'] = bboxes

        return ret


class RandomVShift:
    def __init__(self, ignore_overlap=True, shift_class=None):
        self.ignore_overlap = ignore_overlap
        self.shift_class = shift_class

    def __call__(self, image, bboxes, classes=None, **kwargs):
        # check and select bbox
        new_bboxes = np.array(bboxes, dtype=int)
        if classes is not None:
            classes = np.array(classes)

        iou = Overlap.line(new_bboxes[:, (1, 3)], new_bboxes[:, (1, 3)])
        _ = list(range(len(iou)))
        iou[_, _] = False
        flag = np.any(iou, axis=1) | np.any(iou, axis=0)

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
        non_shift_bbox = new_bboxes[~shift_flag].copy()
        non_shift_delta = np.zeros((len(non_shift_bbox), 2))

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
            non_shift_delta[(non_shift_bbox[:, 1] + non_shift_delta[:, 0] > new_end)] = delta

            new_start = new_b
            old_start = b

        shift_img[new_start:] = image[old_start:]
        non_shift_bbox[:, (1, 3)] += non_shift_delta.astype(int)
        new_bboxes[shift_flag] = shift_bbox
        new_bboxes[~shift_flag] = non_shift_bbox

        return dict(
            image=shift_img,
            bboxes=new_bboxes.astype(bboxes.dtype),
        )


class RandomHShift(RandomVShift):
    def __call__(self, image, bboxes, classes=None, **kwargs):
        ret = dict(image=image, bboxes=bboxes, classes=classes)
        rotate = SimpleRotate(90)
        ret.update(rotate(**ret))
        ret.update(super().__call__(**ret))
        rotate = SimpleRotate(-90)
        ret.update(rotate(**ret))
        ret.pop(rotate.name)

        return ret


class SimpleRotate:
    """only rotates the image by a multiple of angle, and could be change the shape of image"""

    def __init__(self, angle=90):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.angle = angle

    def get_add_params(self, h, w):
        k = self.get_params()
        return {self.name: dict(k=k, h=h, w=w)}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['k'], info['h'], info['w']

    def get_params(self):
        return round(self.angle / 90) % 4

    def __call__(self, image, bboxes=None, **kwargs):
        h, w = image.shape[:2]
        add_params = self.get_add_params(h, w)

        return {
            'image': self.apply_image(image, add_params),
            'bboxes': self.apply_bboxes(bboxes, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        k, _, _ = self.parse_add_params(ret)
        image = np.rot90(image, k)
        image = np.ascontiguousarray(image)
        return image

    def apply_bboxes(self, bboxes, ret):
        if bboxes is not None:
            k, h, w = self.parse_add_params(ret)
            if k == 1:
                bboxes = bboxes[:, (1, 2, 3, 0)]
                bboxes[:, 1::2] = h - bboxes[:, 1::2]
            elif k == 2:
                bboxes = bboxes[:, (2, 3, 0, 1)]
                bboxes[:, 1::2] = h - bboxes[:, 1::2]
                bboxes[:, 0::2] = w - bboxes[:, 0::2]
            elif k == 3:
                bboxes = bboxes[:, (3, 0, 1, 2)]
                bboxes[:, 0::2] = w - bboxes[:, 0::2]

        return bboxes


class Rotate:
    """Rotates the image by angle without changing the shape of image.
    See Also `albumentations.Rotate`

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
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.angle = angle
        self.interpolation = interpolation_mode[interpolation]
        self.expand = expand
        self.center = center
        self.fill = fill

    def get_add_params(self, h, w):
        angle, center, M = self.get_params(w, h)
        return {self.name: dict(angle=angle, center=center, M=M, h=h, w=w)}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['angle'], info['center'], info['M'], info['h'], info['w']

    def get_params(self, w, h):
        angle = self.angle
        center = self.center or (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1)

        return angle, center, M

    @staticmethod
    def transform(x, y, matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    def __call__(self, image, bboxes=None, **kwargs):
        h, w, c = image.shape
        add_params = self.get_add_params(h, w)

        return {
            'image': self.apply_image(image, add_params),
            'bboxes': self.apply_bboxes(bboxes, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        angle, center, M, h, w = self.parse_add_params(ret)

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
        return image

    def apply_bboxes(self, bboxes, ret):
        if bboxes is not None:
            angle, center, M, h, w = self.parse_add_params(ret)
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

        return bboxes


class RandomRotate(Rotate):
    """See Also `torchvision.transforms.RandomRotation`"""

    def get_params(self, w, h):
        if isinstance(self.angle, numbers.Number):
            angle = int(np.random.uniform(-self.angle, self.angle))
        else:
            angle = int(np.random.uniform(*self.angle))

        center = self.center or (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        return angle, center, M


class Affine:
    """rotate, scale and shift the image"""

    def __init__(self, angle=45, translate=(0., 0.), interpolation=0, fill_type=0, fill=(114, 114, 114), scale=1., shear=0.):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.angle = angle
        self.translate = translate
        self.interpolation = interpolation_mode[interpolation]
        self.fill_type = fill_mode[fill_type]
        self.fill = fill
        self.scale = scale
        self.shear = shear

    def get_add_params(self, h, w):
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
        return {self.name: dict(M=M, dst_w=dst_w, dst_h=dst_h)}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['M'], info['dst_w'], info['dst_h']

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
        add_params = self.get_add_params(h, w)

        return {
            'image': self.apply_image(image, add_params),
            'bboxes': self.apply_bboxes(bboxes, add_params),
            **kwargs
        }

    def apply_image(self, image, ret):
        M, dst_w, dst_h = self.parse_add_params(ret)
        return cv2.warpAffine(
            image,
            M, (dst_w, dst_h),
            flags=self.interpolation,
            borderMode=self.fill_type,
            borderValue=self.fill
        )

    def apply_bboxes(self, bboxes, ret):
        if bboxes is not None:
            M, dst_w, dst_h = self.parse_add_params(ret)
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

        return bboxes


class RandomAffine(Affine):
    """see also `torchvision.transforms.RandomAffine` or `albumentations.Affine`"""

    def get_params(self):
        if isinstance(self.angle, numbers.Number):
            angle = int(np.random.uniform(-self.angle, self.angle))
        else:
            angle = int(np.random.uniform(*self.angle))

        if isinstance(self.translate, numbers.Number):
            translate = np.array([-self.translate, self.translate])
        else:
            translate = np.array(self.translate)

        translate = np.random.uniform(*translate, size=2).astype(int)

        return angle, translate, self.scale, self.shear


class Perspective:
    def __init__(self, distortion=0.25, pos=(LEFT, TOP), end_points=None,
                 interpolation=0, fill_type=0, fill=(114, 114, 114), keep_shape=True):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.distortion = distortion
        self.pos = [pos] if isinstance(pos[0], int) else pos
        self.end_points = end_points
        self.interpolation = interpolation_mode[interpolation]
        self.fill_type = fill_mode[fill_type]
        self.fill = fill
        self.keep_shape = keep_shape

    def get_add_params(self, h, w):
        end_points, M = self.get_params(h, w)
        return {self.name: dict(end_points=end_points, M=M, h=h, w=w)}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['end_points'], info['M'], info['h'], info['w']

    def get_params(self, h, w):
        start_points = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=np.float32)

        if self.end_points is None:
            offset_h = self.distortion * h
            offset_w = self.distortion * w

            lt, rt, rd, ld = start_points

            for pos in self.pos:
                if pos == (LEFT, TOP):
                    lt = (offset_w, offset_h)
                elif pos == (RIGHT, TOP):
                    rt = (w - offset_w, offset_h)
                elif pos == (RIGHT, DOWN):
                    rd = (w - offset_w, h - offset_h)
                elif pos == (LEFT, DOWN):
                    ld = (offset_w, h - offset_h)

            end_points = np.array([lt, rt, rd, ld], dtype=np.float32)
        else:
            end_points = self.end_points

        M = cv2.getPerspectiveTransform(start_points, end_points)

        return end_points, M

    def __call__(self, image, bboxes=None, **kwargs):
        h, w = image.shape[:2]
        add_params = self.get_add_params(h, w)

        return {
            'image': self.apply_image(image, add_params),
            'bboxes': self.apply_bboxes(bboxes, add_params),
            **kwargs
        }

    def apply_image(self, image, ret):
        end_points, M, h, w = self.parse_add_params(ret)
        image = cv2.warpPerspective(
            image,
            M, (w, h),
            flags=self.interpolation,
            borderMode=self.fill_type,
            borderValue=self.fill
        )

        if not self.keep_shape:
            rect = cv2.minAreaRect(end_points)
            bbox = cv2.boxPoints(rect).astype(dtype=np.int)
            max_x, max_y = bbox[:, 0].max(), bbox[:, 1].max()
            min_x, min_y = bbox[:, 0].min(), bbox[:, 1].min()
            min_x, min_y = max(min_x, 0), max(min_y, 0)
            image = image[min_y:max_y, min_x:max_x]

        return image

    def apply_bboxes(self, bboxes, ret):
        if bboxes is not None:
            end_points, M, h, w = self.parse_add_params(ret)
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
        return bboxes


class RandomPerspective(Perspective):
    """see also `torchvision.transforms.RandomPerspective` or `albumentations.Perspective`"""

    def get_params(self, h, w):
        if isinstance(self.distortion, numbers.Number):
            distortion = np.array([-self.distortion, self.distortion])
        else:
            distortion = np.array(self.distortion)

        _offset_h = distortion * h
        _offset_w = distortion * w

        offset_h = np.random.uniform(*_offset_h, size=4)
        offset_w = np.random.uniform(*_offset_w, size=4)

        start_points = np.array([(0, 0), (w, 0), (w, h), (0, h)], dtype=np.float32)
        end_points = np.array([
            (offset_w[0], offset_h[0]),
            (w - offset_w[1], offset_h[1]),
            (w - offset_w[2], h - offset_h[2]),
            (offset_w[3], h - offset_h[3])
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(start_points, end_points)

        return end_points, M
