import numpy as np


class Apply:
    """running the func orderly

    Args:
        funcs(list):
        full_result: whether returning total ret from each funcs or not
            if true, return a list
            if false, return a dict
        replace: whether replacing the request used last func return

    Examples
        .. code-block:: python

            from cv_data_parse.data_augmentation import crop, geometry, Apply
            from cv_data_parse.data_augmentation import RandomApply

            image = np.random.rand((256, 256, 3), dtype=np.uint8)
            bboxes = np.random.rand((10, 4))
            ret = dict(image=image, dst=224, bboxes=bboxes)
            ret.update(Apply([crop.Center(), geometry.HFlip()])(**ret))
    """

    def __init__(self, funcs=None, full_result=False, replace=True):
        self.funcs = funcs
        self.full_result = full_result
        self.replace = replace

    def __call__(self, **kwargs):
        ret = kwargs
        full_result = []

        for func in self.funcs:
            r = func(**ret)

            if self.full_result:
                full_result.append(r)

            if self.replace:
                ret.update(r)

        if full_result:
            ret = full_result

        return ret

    def apply_image(self, image, ret):
        for func in self.funcs:
            image = func.apply_image(image, ret)
        return image

    def restore(self, ret):
        if self.full_result:
            _ret = []
            for r in ret:
                for func in self.funcs[::-1]:
                    r = func.restore(r)
                _ret.append(r)

            ret = _ret

        else:
            for func in self.funcs[::-1]:
                ret = func.restore(ret)

        return ret


class RandomApply:
    """running the func with the prob of each func

    Args:
        funcs(list):
        probs(list): running prob of each funcs, default 0.5 to each func
        full_result: whether returning total ret from each funcs or not
            if true, return a list
            if false, return a dict
        replace: whether replacing the request used last func return

    Examples
        .. code-block:: python

            from cv_data_parse.data_augmentation import crop, geometry, RandomApply
            from cv_data_parse.data_augmentation import RandomApply

            image = np.random.rand((256, 256, 3), dtype=np.uint8)
            bboxes = np.random.rand((10, 4))
            ret = dict(image=image, dst=224, bboxes=bboxes)
            ret.update(RandomApply([crop.Center(), geometry.HFlip()])(**ret))
    """

    def __init__(self, funcs=None, probs=None, full_result=False, replace=True):
        self.funcs = funcs
        self.probs = probs
        self.full_result = full_result
        self.replace = replace

    def __call__(self, **kwargs):
        funcs = self.funcs
        probs = self.probs or [0.5] * len(funcs)
        ret = kwargs
        full_result = []
        apply_func_idx = []
        for i, (func, probs) in enumerate(zip(funcs, probs)):
            r = {}

            if np.random.random() < probs:
                r.update(func(**ret))
                apply_func_idx.append(i)

            if self.full_result:
                full_result.append(r)

            if self.replace:
                ret.update(r)

        if self.full_result:
            ret = full_result
        ret['RandomApply'] = apply_func_idx
        return ret

    def apply_image(self, image, ret):
        apply_func_idx = ret['RandomApply']
        for idx in apply_func_idx:
            image = self.funcs[idx].apply_image(image, ret)
        return image

    def restore(self, ret):
        if self.full_result:
            _ret = []
            for r in ret:
                for idx in r['RandomApply'][::-1]:
                    func = self.funcs[idx]
                    r = func.restore(r)
                _ret.append(r)

            ret = _ret

        else:
            for idx in ret['RandomApply'][::-1]:
                func = self.funcs[idx]
                ret = func.restore(ret)

        return ret


class RandomChoice:
    """random choice to run the func

    Args:
        funcs(list):
        probs(list): choice prob of each funcs, default 0.5 to each func
        full_result: whether returning total ret from each funcs or not
            if true, return a list
            if false, return a dict
        replace: whether replacing the request used last func return

    Examples
        .. code-block:: python

            from cv_data_parse.data_augmentation import crop, geometry, RandomChoice
            from cv_data_parse.data_augmentation import RandomApply

            image = np.random.rand((256, 256, 3), dtype=np.uint8)
            bboxes = np.random.rand((10, 4))
            ret = dict(image=image, dst=224, bboxes=bboxes)
            ret.update(RandomChoice([crop.Center(), geometry.HFlip()])(**ret))
    """

    def __init__(self, funcs=None, probs=None, full_result=False, replace=True, choice_size=None):
        self.funcs = funcs
        self.probs = probs or np.ones(len(funcs)) / len(funcs)
        self.full_result = full_result
        self.replace = replace
        self.choice_size = choice_size

    def __call__(self, **kwargs):
        funcs = self.funcs
        probs = self.probs
        ret = kwargs
        full_result = []

        tmp = [(func, probs) for func, probs in zip(funcs, probs)]

        choice_size = self.choice_size or len(tmp)
        func_arg = np.random.choice(range(len(tmp)), size=choice_size, replace=False, p=probs)

        for i in func_arg:
            func, probs = tmp[i]
            r = func(**ret)

            if self.full_result:
                full_result.append(r)

            if self.replace:
                ret.update(r)

        if full_result:
            ret = full_result

        ret.update(func_arg=func_arg)

        return ret

    def restore(self, ret):
        func_arg = ret['func_arg']

        if self.full_result:
            _ret = []
            for r in ret:
                for idx in func_arg[::-1]:
                    func = self.funcs[idx]
                    r = func.restore(r)
                _ret.append(r)

            ret = _ret

        else:
            for idx in func_arg[::-1]:
                func = self.funcs[idx]
                ret = func.restore(ret)

        return ret
