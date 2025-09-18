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

    @staticmethod
    def lexical_matching_score(arr1, arr2, tokens1, tokens2, filter_tokens=()):
        def make_weights(arr, tokens):
            weights = []
            for a, token in zip(arr, tokens):
                result = {}
                for w, idx in zip(a, token):
                    if idx not in filter_tokens and w > 0:
                        idx = str(idx)
                        result.setdefault(idx, w)
                        if w > result[idx]:
                            result[idx] = w

                weights.append(result)
            return weights

        def count_scores(ww1, ww2):
            s = 0
            for token, weight in ww1.items():
                if token in ww2:
                    s += weight * ww2[token]
            return s

        w1 = make_weights(arr1, tokens1)
        w2 = make_weights(arr2, tokens2)
        scores = [[count_scores(ww1, ww2) for ww2 in w2] for ww1 in w1]
        return scores
