"""change the channels of image"""
import numpy as np


class HWC2CHW:
    def __call__(self, image, **kwargs):
        image = np.transpose(image, (2, 0, 1))
        image = np.ascontiguousarray(image)

        return dict(
            image=image
        )
