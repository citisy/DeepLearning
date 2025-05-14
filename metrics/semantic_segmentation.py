import numpy as np


class MaskArea:
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
