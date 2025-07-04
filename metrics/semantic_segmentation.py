import numpy as np
from . import object_detection


class MaskArea:
    @staticmethod
    def areas(mask):
        """Area(boxes)

        Args:
            mask(np.array): shape=(N, H, W)

        Returns:
            areas(np.array): shape=(N,)
        """
        mask = mask.astype(bool)
        return np.sum(mask, axis=(1, 2))

    @staticmethod
    def intersection_areas(mask1, mask2):
        """Area(boxes1 & boxes2)

        Args:
            mask1(np.array): shape=(N, H, W)
            mask2(np.array): shape=(M, H, W)

        Returns:
            intersection_box(np.array): shape=(N, M)
        """
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)
        return np.sum(np.logical_and(mask1[:, None], mask2[None, :]), axis=(2, 3))

    @staticmethod
    def union_areas(mask1, mask2, *args, **kwargs):
        """Area(boxes1 | boxes2)

        Arguments:
            mask1(np.array): shape=(N, H, W)
            mask2(np.array): shape=(M, H, W)

        Returns:
            union_areas(np.array): shape=(N, M)
        """
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)
        return np.sum(np.logical_or(mask1[:, None], mask2[None, :]), axis=(2, 3))

    @staticmethod
    def outer_areas(mask1, mask2, *args, **kwargs):
        """outer rectangle area
        Area(boxes1 | boxes2) - Area(boxes1 & boxes2)

        Arguments:
            mask1(np.array): shape=(N, H, W)
            mask2(np.array): shape=(M, H, W)

        Returns:
            union_areas(np.array): shape=(N, M)
        """
        mask1 = mask1.astype(bool)
        mask2 = mask2.astype(bool)
        return np.sum(np.logical_xor(mask1[:, None], mask2[None, :]), axis=(2, 3))


class Iou(object_detection.Iou):
    def __init__(self, **kwargs):
        super().__init__(area_method=MaskArea, **kwargs)
