import numpy as np


class Apply:
    def __init__(self, funcs=None):
        self.funcs = funcs

    def __call__(self, x, *args, **kwargs):
        for func in self.funcs:
            x = func(x, *args, **kwargs)

        return x


class RandomApply:
    """running the func with the prob of each func

    Args:
        funcs(list):
        probs(list): running prob of each funcs, default 0.5 to each func

    Examples
        .. code-block:: python
            from cv_data_parse.data_augmentation.crop import Corner, Center
            from cv_data_parse.data_augmentation import RandomApply

            x = np.zeros((256, 256, 3), dtype=np.uint8)
            x = RandomApply([Corner(), Center()])(x, 224)
    """

    def __init__(self, funcs=None, probs=None):
        self.funcs = funcs
        self.probs = probs

    def __call__(self, x, *args, **kwargs):
        funcs = self.funcs
        probs = self.probs or [0.5] * len(funcs)

        for func, probs in zip(funcs, probs):
            if np.random.random() < probs:
                x = func(x, *args, **kwargs)

        return x


class RandomChoice:
    """random choice to run the func

    Args:
        funcs(list):
        probs(list): choice prob of each funcs, default 0.5 to each func

    Examples
        .. code-block:: python
            from cv_data_parse.data_augmentation.crop import Corner, Center
            from cv_data_parse.data_augmentation import RandomChoice, RandomApply

            x = np.zeros((256, 256, 3), dtype=np.uint8)
            x = RandomChoice([Corner(), Center()])(x, 224)
            x = RandomChoice([RandomApply([Corner()]), RandomApply([Center()])])(x, 224)
    """

    def __init__(self, funcs=None, probs=None):
        self.funcs = funcs
        self.probs = probs

    def __call__(self, x, *args, **kwargs):
        funcs = self.funcs
        probs = self.probs or [0.5] * len(funcs)

        tmp = [(func, probs) for func, probs in zip(funcs, probs)]

        idx = np.random.choice(range(len(tmp)), size=len(tmp), replace=False, p=probs)

        for i in idx:
            func, probs = tmp[i]
            x = func(x, *args, **kwargs)

        return x
