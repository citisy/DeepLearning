"""change the channels of image, it is best to apply them in final"""
import numpy as np
import cv2


class HWC2CHW:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image)
        return image


class CHW2HWC:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        image = np.transpose(image, (1, 2, 0))
        image = np.ascontiguousarray(image)
        return image


class BGR2RGB:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        image = image.copy()
        image[(0, 1, 2)] = image[(2, 1, 0)]
        return image


class Gray2BGR:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image


class BGR2Gray:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, None]
        return image


class Keep3Dims:
    """input an array which have any(2, 3 or 4) dims, output an array which have 3-dims"""
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        shape = image.shape
        if len(shape) == 2:
            image = image[:, :, None]
        elif len(shape) == 4:
            pass
        return image


class AddXY:
    def __init__(self, axis=-1):
        # y = -cot(x)
        # y in (-1.9, 1.4) where x in (0.5, 2.5)
        self.func = lambda x: (-1 / np.tan(np.linspace(0.5, 2.5, x)) + 1.9) / 3.3 * 255
        self.axis = axis

    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image, *args):
        if self.axis in (-1, 3):
            h, w, c = image.shape
        elif self.axis == 0:
            c, h, w = image.shape
        else:
            raise ValueError(f'Do not support axis = {self.axis}')

        x, y = self.func(w), self.func(h)
        xv, yv = np.meshgrid(x, y)  # (h, w)
        xv = np.expand_dims(xv, axis=self.axis).astype(np.uint8)
        yv = np.expand_dims(yv, axis=self.axis).astype(np.uint8)
        img_xy = np.concatenate((image, xv, yv), axis=self.axis)
        return img_xy

    def restore(self, ret):
        if self.axis in (-1, 3):
            ret['image'] = ret['image'][:, :, :3]
        elif self.axis == 0:
            ret['image'] = ret['image'][:3]

        return ret
