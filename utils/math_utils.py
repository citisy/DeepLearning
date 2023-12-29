import numpy as np


def argsort_and_unique(x, thres=0, keep_small=True):
    """sort the 1-D araay and ignore the slim difference
    Args:
        x: 1-D array
        thres: merge values if the difference of two values lower than the thres
        keep_small: if two values are merged, keep the large one

    Returns:
        order: sorted order of x
        y: x after sorted

    Usage:
        >>> # just sort the values
        >>> x = [1, 2, 3, 4, 5]
        >>> order, _ = argsort_and_unique(x)
        >>> order
        [0 1 2 3 4]

        >>> # merge values if the difference of two values lower than the thres
        >>> # and then, sorted them
        >>> order, y = argsort_and_unique(x, thres=1)
        >>> order
        [0 0 1 1 2]
        >>> y   # 2, 4 will be ignored
        [1 3 5]

        >>> # if two values are merged, keep the large one
        >>> _, y = argsort_and_unique(x, thres=1, keep_small=False)
        >>> y   # 'cause keep the bigger item, 1, 3 will be ignored
        [2 4 5]

        >>> x = [7, 5, 2, 1, 4]
        >>> order, y = argsort_and_unique(x, thres=1, keep_small=False)
        >>> order
        [2 1 0 0 1]
        >>> y
        [2 5 7]

    """
    x = np.array(x)
    arg = np.argsort(np.argsort(x))
    x = np.sort(x)

    order = []
    y = []
    i = 0
    while x.size:
        diff = x[1:] - x[0]
        keep = diff <= thres
        keep = np.append(True, keep)

        tmp = x[keep]

        if keep_small:
            y.append(tmp[0])
        else:
            y.append(tmp[-1])

        order += [i] * len(tmp)
        x = x[~keep]
        i += 1

    order = np.array(order)[arg]
    y = np.array(y)
    return order, y


def arg_order_sort_2D(x, key=None, **kwargs):
    """sort the 2-D araay following the columns index, different to `np.argsort()`
    Args:
        x (np.ndarray): 2-D array, (m, n)
        key (tuple): indexs fall in [0, n)
        **kwargs: kwargs for `argsort_and_unique()`

    Returns:
        arg (np.ndarray): 1-D array (m, )

    Usage:
        >>> # sort the values by all the order keys
        >>> x = np.array([[1, 2, 3, 5, 4], [10, 9, 8, 6, 7]]).T
        >>> arg_order_sort_2D(x)
        [0 1 2 4 3]

        >>> # sort by x[:, 1] firstly; sort by x[:, 0] secondly
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
    """sort the 2-D araay following the columns index, different to `np.sort()`
    Args:
        x (np.ndarray): 2-D array, (m, n)
        key (tuple): indexs fall in [0, n)
        **kwargs: kwargs for `argsort_and_unique()`

    Returns:
        y (np.ndarray): 2-D array (m, n), x after sorted

    Usage:
        >>> # sort the values by all the order keys
        >>> x = np.array([[1, 2, 3, 5, 4], [10, 9, 8, 6, 7]]).T
        >>> order_sort_2D(x).T
        [[ 1  2  3  4  5]
         [10  9  8  7  6]]

        >>> # sort by x[:, 1] firstly, sort by x[:, 0] secondly
        >>> order_sort_2D(x, key=(1, 0)).T
        [[ 5  4  3  2  1]
        [ 6  7  8  9 10]]

    """
    arg = arg_order_sort_2D(x, key=key, **kwargs)
    return x[arg]


def transpose(x):
    """transpose the list, same behaviour to `np.transpose()`
    Args:
        x (List[list]): 2-D list

    Usage:

    """
    return list(zip(*x))
