"""change the pixel of image"""
import cv2
import numbers
import numpy as np
from . import RandomChoice
from metrics.object_detection import Iou


class MinMax:
    """[0, 255] to [0, 1]"""

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        return image / 255.

    def restore(self, ret):
        ret['image'] = ret['image'] * 255
        return ret


class Clip:
    def __init__(self, a_min=0, a_max=255):
        self.a_min = a_min
        self.a_max = a_max

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        return np.clip(image, self.a_min, self.a_max)


class GaussNoise:
    """添加高斯噪声

    Args:
        mean: 高斯分布的均值
        sigma: 高斯分布的标准差
    """

    def __init__(self, mean=0, sigma=25):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        gauss = np.random.normal(self.mean, self.sigma, image.shape)
        image = (image + gauss).clip(min=0, max=255).astype(image.dtype)

        return image


class SaltNoise:
    """添加椒盐噪声

    Args:
        s_vs_p: 添加椒盐噪声的数目比例
        amount: 添加噪声图像像素的数目
    """

    def __init__(self, s_vs_p=0.5, amount=0.04):
        self.s_vs_p = s_vs_p
        self.amount = amount

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        image = np.copy(image)
        num_salt = np.ceil(self.amount * image.size * self.s_vs_p)
        coords = tuple(np.random.randint(0, i - 1, int(num_salt)) for i in image.shape)
        image[coords] = 255
        num_pepper = np.ceil(self.amount * image.size * (1. - self.s_vs_p))
        coords = tuple(np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape)
        image[coords] = 0
        return image


class PoissonNoise:
    """添加泊松噪声"""

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        image = np.random.poisson(image * vals) / float(vals)

        return image


class SpeckleNoise:
    """添加斑点噪声"""

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        gauss = np.random.randn(*image.shape)
        noisy_image = image + image * gauss
        image = np.clip(noisy_image, a_min=0, a_max=255)
        return image


class Normalize:
    """Normalizes a ndarray image or image with mean and standard deviation.
    See Also `torchvision.transforms.Normalize`
    """

    def __init__(self, mean=None, std=None):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.mean = mean
        self.std = std

    def get_params(self, image):
        mean = self.mean if self.mean is not None else np.mean(image, axis=(0, 1))
        std = self.std if self.std is not None else np.std(image, axis=(0, 1))

        return mean, std

    def get_add_params(self, image):
        mean, std = self.get_params(image)
        return {self.name: dict(mean=mean, std=std)}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['mean'], info['std']

    def __call__(self, image, **kwargs):
        add_params = self.get_add_params(image)

        return {
            'image': self.apply_image(image, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        mean, std = self.parse_add_params(ret)
        return (image - mean) / std

    def restore(self, ret):
        mean, std = self.parse_add_params(ret)
        image = ret['image']
        ret['image'] = image * std + mean
        return ret


class Pca:
    """after dimensionality reduction by pca
    add the random scale factor"""

    def __init__(self, eigenvectors=None, eigen_values=None):
        self.eigenvectors = eigenvectors
        self.eigen_values = eigen_values

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        a = np.random.normal(0, 0.1)

        noisy_image = np.array(image, dtype=float)

        if self.eigenvectors is None:
            eigenvectors = []
            eigen_values = []

            for i in range(noisy_image.shape[-1]):
                x = noisy_image[:, :, i]

                for j in range(x.shape[0]):
                    n = np.mean(x[j])
                    s = np.std(x[j], ddof=1)
                    x[j] = (x[j] - n) / s

                cov = np.cov(x, rowvar=False)

                # todo: Complex return, is that wrong?
                eigen_value, eigen_vector = np.linalg.eig(cov)

                # arg = np.argsort(eigen_value)[::-1]
                # eigen_vector = eigen_vector[:, arg]
                # eigen_value = eigen_value[arg]

                eigen_values.append(eigen_value)
                eigenvectors.append(eigen_vector)

        for i in range(image.shape[-1]):
            noisy_image[:, :, i] = noisy_image[:, :, i] @ (self.eigenvectors[i].T * self.eigen_values[i] * a).T

        return noisy_image


class AdjustHsv:
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return image


class AdjustBrightness:
    """Adjusts brightness of an image.
    See Also `torchvision.transforms.functional.adjust_brightness` or `albumentations.adjust_brightness_torchvision`

    Args:
        offset (float):
            factor = 1 + offset
            factor in [0, inf], where 0 give black, 1 give original, 2 give brightness doubled

    """

    def __init__(self, offset=.5):
        self.offset = offset

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        factor = max(1 + self.offset, 0)
        table = np.array([i * factor for i in range(0, 256)]).clip(0, 255).astype('uint8')
        image = cv2.LUT(image, table)
        return image


class AdjustContrast:
    """Adjusts contrast of an image.
    See Also `torchvision.transforms.functional.adjust_contrast` or `albumentations.adjust_contrast_torchvision`

    Args:
        offset (float):
            factor = 1 + offset
            factor in [0, inf], where 0 give solid gray, 1 give original, 2 give contrast doubled

    """

    def __init__(self, offset=.5):
        self.offset = offset

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        factor = max(1 + self.offset, 0)
        table = np.array([(i - 74) * factor + 74 for i in range(0, 256)]).clip(0, 255).astype('uint8')
        image = cv2.LUT(image, table)
        return image


class AdjustSaturation:
    """Adjusts color saturation of an image.
    See Also `torchvision.transforms.functional.adjust_saturation` or `albumentations.adjust_saturation_torchvision`

    Args:
        offset (float):
            factor = 1 + offset
            factor, where 0 give black and white, 1 give original, 2 give saturation doubled

    """

    def __init__(self, offset=.5):
        self.offset = offset

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        factor = 1 + self.offset
        dtype = image.dtype
        image = image.astype(np.float32)
        alpha = np.random.uniform(max(0, 1 - factor), 1 + factor)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image[..., np.newaxis]
        image = image * alpha + gray_image * (1 - alpha)
        image = image.clip(0, 255).astype(dtype)
        return image


class AdjustHue:
    """Adjusts hue of an image.
    See Also `torchvision.transforms.functional.adjust_hue` or `albumentations.adjust_hue_torchvision`

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    Args:
        offset (float):
            factor = 1 + offset
            factor in [-0.5, 0.5], where
            -0.5 give complete reversal of hue channel in HSV space in positive direction respectively
            0 give no shift,
            0.5 give complete reversal of hue channel in HSV space in negative direction respectively

    Returns:
        np.array: Hue adjusted image.

    """

    def __init__(self, offset=.1):
        self.offset = offset

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        factor = min(max(1 + self.offset, -0.5), 0.5)

        dtype = image.dtype
        image = image.astype(np.uint8)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        h, s, v = cv2.split(hsv_image)

        alpha = np.random.uniform(factor, factor)
        h = h.astype(np.uint8)
        # uint8 addition take cares of rotation across boundaries
        with np.errstate(over="ignore"):
            h += np.uint8(alpha * 255)
        hsv_image = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR_FULL).astype(dtype)
        return image


class Jitter:
    """Randomly change the brightness, contrast, saturation and hue of an image.
    See Also `torchvision.transforms.ColorJitter` or `albumentations.ColorJitter`

    Args:
        apply_func: (brightness, contrast, saturation, hue)
        offsets (tuple): offset of apply_func

    """

    def __init__(
            self,
            offsets=(0.5, 0.5, 0.5, 0.1),
            apply_func=(AdjustBrightness, AdjustContrast, AdjustSaturation, AdjustHue)
    ):
        funcs = [func(offset) for offset, func in zip(offsets, apply_func)]
        self.apply = RandomChoice(funcs)

    def __call__(self, image, **kwargs):
        return self.apply(image=image)


class GaussianBlur:
    """see also `torchvision.transforms.GaussianBlur` or `albumentations.GaussianBlur`"""

    def __init__(self, ksize=None, sigma=(.5, .5)):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.ksize = ksize
        self.sigma = sigma

    def get_params(self, w, h):
        ksize = self.ksize
        if not ksize:
            ksize = min(w, h) // 8
            ksize = (ksize * 2) + 1

        return ksize

    def get_add_params(self, w, h):
        ksize = self.get_params(w, h)
        return {self.name: dict(ksize=ksize)}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['ksize']

    def __call__(self, image, **kwargs):
        h, w, c = image.shape
        add_params = self.get_add_params(w, h)

        return {
            'image': self.apply_image(image, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        ksize = self.parse_add_params(ret)
        return cv2.GaussianBlur(image, ksize, sigmaX=self.sigma[0], sigmaY=self.sigma[1])


class MotionBlur:
    """see also `albumentations.MotionBlur`"""

    def __init__(self, degree=12, angle=90):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.degree = degree
        self.angle = angle

    def get_params(self):
        return self.degree, self.angle

    def get_add_params(self):
        degree, angle = self.get_params()
        return {self.name: dict(
            degree=degree,
            angle=angle
        )}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['degree'], info['angle']

    def __call__(self, image, degree=None, angle=None, **kwargs):
        add_params = self.get_add_params()

        return {
            'image': self.apply_image(image, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        degree, angle = self.parse_add_params(ret)
        M = cv2.getRotationMatrix2D((degree // 2, degree // 2), angle, 1)
        motion_blur_kernel = np.zeros((degree, degree))
        motion_blur_kernel[degree // 2, :] = 1
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        image = cv2.filter2D(image, -1, motion_blur_kernel)
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image


class RandomMotionBlur(MotionBlur):
    def get_params(self):
        degree = int((np.random.beta(4, 4) - 0.5) * 2 * self.degree)

        if isinstance(self.angle, numbers.Number):
            angle = int(np.random.uniform(-self.angle, self.angle))
        else:
            angle = int(np.random.uniform(*self.angle))

        return degree, angle


class Erase:
    def __init__(self, scale=0.1, ratio=1., fill=None, max_iter=10):
        self.name = __name__.split('.')[-1] + '.' + self.__class__.__name__
        self.scale = scale
        self.ratio = ratio
        self.fill = fill if fill is not None else np.random.randint(100, 125, size=3)
        self.max_iter = max_iter

    def get_params(self):
        scales = [self.scale for _ in range(self.max_iter)]
        ratios = [self.ratio for _ in range(self.max_iter)]
        return scales, ratios

    def get_add_params(self):
        scales, ratios = self.get_params()
        return {self.name: dict(
            scales=scales,
            ratios=ratios
        )}

    def parse_add_params(self, ret):
        info = ret[self.name]
        return info['scales'], info['ratios']

    def __call__(self, image, **kwargs):
        add_params = self.get_add_params()

        return {
            'image': self.apply_image(image, add_params),
            **add_params
        }

    def apply_image(self, image, ret):
        scales, ratios = self.parse_add_params(ret)
        h, w, c = image.shape
        area = h * w
        for scale, ratio in zip(scales, ratios):
            erase_area = area * scale

            _h = int(round(np.sqrt(erase_area * ratio)))
            _w = int(round(np.sqrt(erase_area / ratio)))

            if _w < w and _h < h:
                y1 = np.random.randint(0, h - _h)
                x1 = np.random.randint(0, w - _w)
                if c == 3:
                    image[y1:y1 + _h, x1:x1 + _w] = self.fill

                break

        return image


class RandomErasing(Erase):
    """https://arxiv.org/abs/1708.04896
    see also `torchvision.transforms.RandomErasing`"""

    def get_params(self):
        scales, ratios = [], []
        for _ in range(self.max_iter):
            if isinstance(self.scale, numbers.Number):
                scale = np.random.uniform(self.scale - 0.05, self.scale + 0.05)
            else:
                scale = np.random.uniform(*self.scale)

            if isinstance(self.ratio, numbers.Number):
                ratio = np.random.uniform(self.ratio - 0.5, self.ratio + 0.5)
            else:
                ratio = np.random.uniform(*self.ratio)
            scales.append(scale)
            ratios.append(ratio)

        return scales, ratios


class CutOut:
    """https://arxiv.org/abs/1708.04552
    See Also `albumentations.Cutout`
    """

    def __init__(self, scales=None, iou_thres=0.6):
        self.scales = scales or [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        self.iou_thres = iou_thres

    def __call__(self, image, bboxes=None, classes=None, **kwargs):
        image, bboxes, classes = self.apply_image_bboxes_classes(image, bboxes, classes)

        return dict(
            image=image,
            bboxes=bboxes,
            classes=classes,
        )

    def apply_image_bboxes_classes(self, image, bboxes, classes):
        h, w = image.shape[:2]

        mask_bboxes = []

        for s in self.scales:
            mask_h = np.random.randint(1, int(h * s))  # create random masks
            mask_w = np.random.randint(1, int(w * s))

            # box
            xmin = max(0, np.random.randint(0, w) - mask_w // 2)
            ymin = max(0, np.random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            image[ymin:ymax, xmin:xmax] = [np.random.randint(64, 191) for _ in range(3)]

            if bboxes is not None and s > 0.03:
                mask_bboxes.append([xmin, ymin, xmax, ymax])

        if bboxes is not None:
            iou = Iou.iou(bboxes, mask_bboxes)
            iou = np.max(iou, axis=1)
            idx = iou < self.iou_thres
            bboxes = bboxes[idx]

            if classes is not None:
                classes = classes[idx]

        return image, bboxes, classes


class AxisProjection:
    def __init__(self, axis=0):
        """

        Args:
            axis: (h, w, c),
                0 gives that all pixels are projected to x-axis,
                1 gives that all pixels are projected to y-axis
        """
        self.axis = axis

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        x = image.mean(axis=self.axis)
        x = np.expand_dims(x, self.axis)
        x = np.repeat(x, image.shape[self.axis], axis=self.axis)
        return x
