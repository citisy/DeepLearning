"""change the channels of image"""
import numpy as np
import cv2


class HWC2CHW:
    def __call__(self, image, **kwargs):
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image)

        return dict(
            image=image
        )


class BGR2RGB:
    def __call__(self, image, **kwargs):
        image = image.copy()
        image[(0, 1, 2)] = image[(2, 1, 0)]
        return dict(
            image=image
        )


class Gray2BGR:
    def __call__(self, image, **kwargs):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        return dict(
            image=image
        )
