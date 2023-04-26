"""change the pixel of image"""
import cv2
import numbers
import numpy as np
from . import RandomChoice


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
        gauss = np.random.normal(self.mean, self.sigma, image.shape)
        noisy_image = image + gauss
        noisy_image = np.clip(noisy_image, a_min=0, a_max=255)

        return dict(
            image=noisy_image
        )


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
        noisy_image = np.copy(image)
        num_salt = np.ceil(self.amount * image.size * self.s_vs_p)
        coords = tuple(np.random.randint(0, i - 1, int(num_salt)) for i in image.shape)
        noisy_image[coords] = 255
        num_pepper = np.ceil(self.amount * image.size * (1. - self.s_vs_p))
        coords = tuple(np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape)
        noisy_image[coords] = 0

        return dict(
            image=noisy_image
        )


class PoissonNoise:
    """添加泊松噪声"""

    def __call__(self, image, **kwargs):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_image = np.random.poisson(image * vals) / float(vals)

        return dict(
            image=noisy_image
        )


class SpeckleNoise:
    """添加斑点噪声"""

    def __call__(self, image, **kwargs):
        gauss = np.random.randn(*image.shape)
        noisy_image = image + image * gauss
        noisy_image = np.clip(noisy_image, a_min=0, a_max=255)

        return dict(
            image=noisy_image
        )


class Normalize:
    """Normalizes a ndarray image or image with mean and standard deviation.
    See Also `torchvision.transforms.Normalize`
    """

    def __call__(self, image, **kwargs):
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1))

        image = (image - mean) / std

        return {
            'image': image,
            'pixel.Normalize': dict(
                mean=mean,
                std=std
            )}


class Pca:
    """after dimensionality reduction by pca
    add the random scale factor"""

    def __init__(self, eigenvectors=None, eigen_values=None):
        self.eigenvectors = eigenvectors
        self.eigen_values = eigen_values

    def __call__(self, image, **kwargs):

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

        return dict(
            image=noisy_image
        )


class AdjustBrightness:
    """Adjusts brightness of an image.
    See Also `torchvision.transforms.functional.adjust_brightness`

    Args:
        offset (float):
            factor = 1 + offset
            factor in [0, inf], where 0 give black, 1 give original, 2 give brightness doubled

    """

    def __init__(self, offset=.5):
        self.offset = offset

    def __call__(self, image, **kwargs):
        factor = max(1 + self.offset, 0)
        table = np.array([i * factor for i in range(0, 256)]).clip(0, 255).astype('uint8')
        image = cv2.LUT(image, table)

        return dict(
            image=image
        )


class AdjustContrast:
    """Adjusts contrast of an image.
    See Also `torchvision.transforms.functional.adjust_contrast`

    Args:
        offset (float):
            factor = 1 + offset
            factor in [0, inf], where 0 give solid gray, 1 give original, 2 give contrast doubled

    """

    def __init__(self, offset=.5):
        self.offset = offset

    def __call__(self, image, **kwargs):
        factor = max(1 + self.offset, 0)
        table = np.array([(i - 74) * factor + 74 for i in range(0, 256)]).clip(0, 255).astype('uint8')
        image = cv2.LUT(image, table)

        return dict(
            image=image
        )


class AdjustSaturation:
    """Adjusts color saturation of an image.
    See Also `torchvision.transforms.functional.adjust_saturation`

    Args:
        offset (float):
            factor = 1 + offset
            factor, where 0 give black and white, 1 give original, 2 give saturation doubled

    """

    def __init__(self, offset=.5):
        self.offset = offset

    def __call__(self, image, **kwargs):
        factor = 1 + self.offset
        dtype = image.dtype
        image = image.astype(np.float32)
        alpha = np.random.uniform(max(0, 1 - factor), 1 + factor)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = gray_image[..., np.newaxis]
        image = image * alpha + gray_image * (1 - alpha)
        image = image.clip(0, 255).astype(dtype)

        return dict(
            image=image
        )


class AdjustHue:
    """Adjusts hue of an image.
    See Also `torchvision.transforms.functional.adjust_hue`

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

        return dict(
            image=image
        )


class Jitter:
    """Randomly change the brightness, contrast, saturation and hue of an image.
    See Also `torchvision.transforms.ColorJitter`

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
    """see also `torchvision.transforms.GaussianBlur`"""

    def __init__(self, ksize=None, sigma=(.5, .5)):
        self.ksize = ksize
        self.sigma = sigma

    def __call__(self, image, **kwargs):
        h, w, c = image.shape

        ksize = self.ksize
        if not ksize:
            ksize = min(w, h) // 8
            ksize = (ksize * 2) + 1

        return {
            'image': cv2.GaussianBlur(image, ksize, sigmaX=self.sigma[0], sigmaY=self.sigma[1]),
            'pixel.GaussianBlur': dict(ksize=ksize)
        }


class MotionBlur:
    def __init__(self, degree=12, angle=90):
        self.degree = degree
        self.angle = angle

    def get_params(self):
        return self.degree, self.angle

    def __call__(self, image, degree=None, angle=None, **kwargs):
        degree, angle = self.get_params()

        M = cv2.getRotationMatrix2D((degree // 2, degree // 2), angle, 1)
        motion_blur_kernel = np.zeros((degree, degree))
        motion_blur_kernel[degree // 2, :] = 1
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree
        image = cv2.filter2D(image, -1, motion_blur_kernel)
        image = np.clip(image, 0, 255).astype(np.uint8)

        return {
            'image': image,
            'pixel.MotionBlur': dict(
                degree=degree,
                angle=angle
            )}


class RandomMotionBlur(MotionBlur):
    def get_params(self):
        degree = int((np.random.beta(4, 4) - 0.5) * 2 * self.degree)
        angle = int(np.random.uniform(-self.angle, self.angle))

        return degree, angle
