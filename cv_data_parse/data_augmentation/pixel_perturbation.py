"""change the pixel of image"""
import cv2
import numpy as np


def add_gauss_noise(image: np.ndarray, mean=0, sigma=25):
    """添加高斯噪声

    Args:
        image: (h, w, c) image
        mean: 高斯分布的均值
        sigma: 高斯分布的标准差
    """
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_img = image + gauss
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)

    return noisy_img


def add_salt_noise(image: np.ndarray, s_vs_p=0.5, amount=0.04):
    """添加椒盐噪声

    Args:
        image: (h, w, c) image
        s_vs_p: 添加椒盐噪声的数目比例
        amount: 添加噪声图像像素的数目
    """
    noisy_img = np.copy(image)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = tuple(np.random.randint(0, i - 1, int(num_salt)) for i in image.shape)
    noisy_img[coords] = 255
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = tuple(np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape)
    noisy_img[coords] = 0

    return noisy_img


def add_poisson_noise(image: np.ndarray):
    """添加泊松噪声

    Args:
        image: (h, w, c) image
    """
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_img = np.random.poisson(image * vals) / float(vals)
    return noisy_img


def add_speckle_noise(image: np.ndarray):
    """添加斑点噪声

    Args:
        image: (h, w, c) image
    """
    gauss = np.random.randn(*image.shape)
    noisy_img = image + image * gauss
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    return noisy_img


def normalize(image):
    """Normalizes a ndarray image or image with mean and standard deviation.
    See Also `torchvision.transforms.Normalize`

    Args:
        image (np.array): (h, w, c) image

    """
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))

    image = (image - mean) / std
    return image


def pca(image, eigenvectors=None, eigen_values=None):
    """after dimensionality reduction by pca
    add the random scale factor"""
    a = np.random.normal(0, 0.1)

    noisy_img = np.array(image, dtype=float)

    if eigenvectors is None:
        eigenvectors = []
        eigen_values = []

        for i in range(noisy_img.shape[-1]):
            x = noisy_img[:, :, i]

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
        noisy_img[:, :, i] = noisy_img[:, :, i] @ (eigenvectors[i].T * eigen_values[i] * a).T

    return noisy_img, (eigenvectors, eigen_values)


def adjust_brightness(img, offset=.5):
    """Adjusts brightness of an image.
    See Also `torchvision.transforms.functional.adjust_brightness`

    Args:
        img (np.array): (h, w, c) image
        offset (float):
            factor = 1 + offset
            factor in [0, inf], where 0 give black, 1 give original, 2 give brightness doubled

    """
    factor = max(1 + offset, 0)
    table = np.array([i * factor for i in range(0, 256)]).clip(0, 255).astype('uint8')

    return cv2.LUT(img, table)


def adjust_contrast(img, offset=.5):
    """Adjusts contrast of an image.
    See Also `torchvision.transforms.functional.adjust_contrast`

    Args:
        img (np.array): (h, w, c) image
        offset (float):
            factor = 1 + offset
            factor in [0, inf], where 0 give solid gray, 1 give original, 2 give contrast doubled

    """
    factor = max(1 + offset, 0)
    table = np.array([(i - 74) * factor + 74 for i in range(0, 256)]).clip(0, 255).astype('uint8')

    return cv2.LUT(img, table)


def adjust_saturation(img, offset=.5):
    """Adjusts color saturation of an image.
    See Also `torchvision.transforms.functional.adjust_saturation`

    Args:
        img (np.array): (h, w, c) image
        offset (float):
            factor = 1 + offset
            factor, where 0 give black and white, 1 give original, 2 give saturation doubled

    """
    factor = 1 + offset
    dtype = img.dtype
    img = img.astype(np.float32)
    alpha = np.random.uniform(max(0, 1 - factor), 1 + factor)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = gray_img[..., np.newaxis]
    img = img * alpha + gray_img * (1 - alpha)
    return img.clip(0, 255).astype(dtype)


def adjust_hue(img, offset=.1):
    """Adjusts hue of an image.
    See Also `torchvision.transforms.functional.adjust_hue`

    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    Args:
        img (np.array): (h, w, c) image
        offset (float):
            factor = 1 + offset
            factor in [-0.5, 0.5], where
            -0.5 give complete reversal of hue channel in HSV space in positive direction respectively
            0 give no shift,
            0.5 give complete reversal of hue channel in HSV space in negative direction respectively

    Returns:
        np.array: Hue adjusted image.

    """
    factor = min(max(1 + offset, -0.5), 0.5)

    dtype = img.dtype
    img = img.astype(np.uint8)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    h, s, v = cv2.split(hsv_img)

    alpha = np.random.uniform(factor, factor)
    h = h.astype(np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        h += np.uint8(alpha * 255)
    hsv_img = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR_FULL).astype(dtype)


def jitter(
        image,
        offsets=(0.5, 0.5, 0.5, 0.1),
        apply_func=(adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue)
):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    See Also `torchvision.transforms.ColorJitter`

    Args:
        image (np.array): (h, w, c) image
        apply_func: (brightness, contrast, saturation, hue)
        offsets (tuple): offset of apply_func

    """
    tmp = {func: offset for offset, func in zip(offsets, apply_func)}
    keys = list(tmp.keys())
    idx = np.random.choice(range(len(keys)), size=len(keys), replace=False)

    for i in idx:
        func = keys[i]
        image = func(image, tmp[func])

    return image
