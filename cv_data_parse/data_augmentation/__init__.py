import numpy as np


def random_apply(
        x, funcs: list,
        probs: list = None,
        func_kwargs: list = None
):
    probs = probs or [0.5] * len(funcs)
    func_kwargs = func_kwargs or [dict()] * len(funcs)

    for func, probs, kwargs in zip(funcs, probs, func_kwargs):
        if np.random.random() < probs:
            x = func(x, **kwargs)

    return x


def random_choice(
        x, funcs: list,
        probs: list = None,
        func_kwargs: list = None
):
    probs = probs or [0.5] * len(funcs)
    func_kwargs = func_kwargs or [dict()] * len(funcs)

    tmp = [(func, probs, kwargs) for func, probs, kwargs in zip(funcs, probs, func_kwargs)]

    idx = np.random.choice(range(len(tmp)), size=len(tmp), replace=False, p=probs)

    for i in idx:
        func, probs, kwargs = tmp[i]
        x = func(x, **kwargs)

    return x
