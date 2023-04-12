import numpy as np


class Apply:
    def __init__(self, funcs=None, full_result=False, replace=True):
        self.funcs = funcs
        self.full_result = full_result
        self.replace = replace

    def __call__(self, image, *args, **kwargs):
        ret = dict(image=image)
        for func in self.funcs:
            r = func(ret['image'], *args, **kwargs)

            if self.full_result:
                if self.replace:
                    ret['image'] = r['image']

                ret.setdefault('full_result', []).append(r)
            else:
                ret.update(r)

        return ret


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
            ret = RandomApply([Corner(), Center()])(x, 224)
    """

    def __init__(self, funcs=None, probs=None, full_result=False):
        self.funcs = funcs
        self.probs = probs
        self.full_result = full_result

    def __call__(self, image, *args, **kwargs):
        funcs = self.funcs
        probs = self.probs or [0.5] * len(funcs)
        ret = dict(image=image)

        for func, probs in zip(funcs, probs):
            r = dict()

            if np.random.random() < probs:
                r = func(ret['image'], *args, **kwargs)

            if self.full_result:
                ret['image'] = r['image']
                ret.setdefault('full_result', []).append(r)
            else:
                ret.update(r)

        return ret


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
            ret = RandomChoice([Corner(), Center()])(x, 224)
            ret = RandomChoice([RandomApply([Corner()]), RandomApply([Center()])])(x, 224)
    """

    def __init__(self, funcs=None, probs=None, full_result=False):
        self.funcs = funcs
        self.probs = probs
        self.full_result = full_result

    def __call__(self, image, *args, **kwargs):
        funcs = self.funcs
        probs = self.probs or [0.5] * len(funcs)

        tmp = [(func, probs) for func, probs in zip(funcs, probs)]

        func_arg = np.random.choice(range(len(tmp)), size=len(tmp), replace=False, p=probs)
        ret = dict(image=image, func_arg=func_arg)

        for i in func_arg:
            func, probs = tmp[i]
            r = func(ret['image'], *args, **kwargs)

            if self.full_result:
                ret['image'] = r['image']
                ret.setdefault('full_result', []).append(r)
            else:
                ret.update(r)

        return ret
