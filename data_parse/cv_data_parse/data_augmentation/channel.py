"""change the channels of image"""
import numpy as np
import cv2


class HWC2CHW:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image):
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image)
        return image


class CHW2HWC:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image):
        image = np.transpose(image, (1, 2, 0))
        image = np.ascontiguousarray(image)
        return image


class BGR2RGB:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image):
        image = image.copy()
        image[(0, 1, 2)] = image[(2, 1, 0)]
        return image


class Gray2BGR:
    def __call__(self, image, **kwargs):
        return dict(
            image=self.apply_image(image)
        )

    def apply_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
