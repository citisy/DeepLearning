import numpy as np


def argsort_and_unique(x, thres=0, keep_small=True):
    """
    Usage:
        >>> x = [1, 2, 3, 4, 5]
        >>> order, _ = argsort_and_unique(x)
        >>> order
        [0 1 2 3 4]

        >>> order, values = argsort_and_unique(x, thres=1)
        >>> order
        [0 0 1 1 2]
        >>> values
        [1 3 5]

        >>> _, values = argsort_and_unique(x, thres=1, keep_small=False)
        >>> values
        [2 4 5]

        >>> x = [7, 5, 2, 1, 4]
        >>> order, values = argsort_and_unique(x, thres=1, keep_small=False)
        >>> order
        [2 1 0 0 1]
        >>> values
        [2 5 7]

    """
    x = np.array(x)
    arg = np.argsort(np.argsort(x))
    x = np.sort(x)

    order = []
    values = []
    i = 0
    while x.size:
        diff = x[1:] - x[0]
        keep = diff <= thres
        keep = np.append(True, keep)

        tmp = x[keep]

        if keep_small:
            values.append(tmp[0])
        else:
            values.append(tmp[-1])

        order += [i] * len(tmp)
        x = x[~keep]
        i += 1

    order = np.array(order)[arg]
    values = np.array(values)
    return order, values


def arg_order_sort_2D(x, key=None, **kwargs):
    """
    Usage:
        >>> x = np.array([[1, 2, 3, 5, 4], [10, 9, 8, 6, 7]]).T
        >>> arg_order_sort_2D(x)
        [0 1 2 4 3]

        >>> order_sort_2D(x, key=(1, 0))
        [3 4 2 1 0]

    """
    x = np.array(x)
    if key is None:
        key = list(range(x.shape[-1]))

    orders = []
    for k in key:
        order, _ = argsort_and_unique(x[:, k], **kwargs)
        orders.append(order)

    orders = np.array(orders).T
    idx = np.array(range(x.shape[0])).reshape((-1, 1))
    orders = np.concatenate([orders, idx], axis=1)
    orders = sorted(orders, key=lambda x: [x[i] for i in range(len(key))])
    orders = np.array(orders)
    arg = orders[:, -1]
    return arg


def order_sort_2D(x, key=None, **kwargs):
    """
    Usage:
        >>> x = np.array([[1, 2, 3, 5, 4], [10, 9, 8, 6, 7]]).T
        >>> order_sort_2D(x).T
        [[ 1  2  3  4  5]
         [10  9  8  7  6]]

        >>> order_sort_2D(x, key=(1, 0)).T
        [[ 5  4  3  2  1]
        [ 6  7  8  9 10]]

    """
    arg = arg_order_sort_2D(x, key=key, **kwargs)
    return x[arg]
