import numpy as np


class Similarity:
    @staticmethod
    def euclidean(arr1, arr2):
        arr1 = arr1[:, None]
        arr2 = arr2[None, :]
        distance = np.linalg.norm(arr1 - arr2, axis=-1)
        return 1 / (1 + distance)

    @staticmethod
    def cosine(arr1, arr2):
        arr1 = arr1[:, None]
        arr2 = arr2[None, :]
        dot_product = np.sum(arr1 * arr2, axis=-1)
        norm_arr1 = np.linalg.norm(arr1, axis=-1)
        norm_arr2 = np.linalg.norm(arr2, axis=-1)
        return dot_product / (norm_arr1 * norm_arr2)
