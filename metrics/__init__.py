import numbers
import difflib


def diff_str(a, b):
    if a != b:
        r = difflib.Differ().compare(a, b)

        i = 0
        j = 0
        ra = []
        rb = []
        tmp_a = []
        tmp_b = []
        for c in r:
            if c.startswith(' '):
                if tmp_a:
                    ra.append((tmp_a[0], i))
                if tmp_b:
                    rb.append((tmp_b[0], j))
                tmp_a = []
                tmp_b = []
                i += 1
                j += 1
            elif c.startswith('-'):
                tmp_a.append(i)
                i += 1
            elif c.startswith('+'):
                tmp_b.append(j)
                j += 1

        return False, (tuple(ra), tuple(rb))

    else:
        return True, ()


def diff_obj(obj1, obj2, num_eps=0, return_str_span=False):
    path = []
    if isinstance(obj1, numbers.Number) and isinstance(obj2, numbers.Number):
        if num_eps:
            flag = abs(obj1 - obj2) < num_eps
        else:
            flag = obj1 == obj2

    elif isinstance(obj1, str) and isinstance(obj2, str):
        if return_str_span:
            flag, (obj1, obj2) = diff_str(obj1, obj2)
        else:
            flag = obj1 == obj2

    else:
        flag = obj1 == obj2

    if not flag:
        path.append((obj1, obj2))

    return flag, path


def diff(obj1, obj2, **kwargs):
    """
    Usages:
        >>> a = {'a': {'b': {'c': 1.1, 'd': [1,2,3], 'e': 'hello', 'f': 4}}}
        >>> b = {'a': {'b': {'c': 1.2, 'd': [1,2,4], 'e': 'hallo'}}}
        >>> flag, path = diff(a, b)
        >>> flag
        False

        >>> for p in path: p
        ['a', 'b', 'c', (1.1, 1.2)]
        ['a', 'b', 'd', 2, (3, 4)]
        ['a', 'b', 'e', ('hello', 'hallo')]
        ['a', 'b', 'f', (4, None)]

        >>> flag, path = diff(a, b, num_eps=0.5)
        >>> for p in path: p
        ['a', 'b', 'd', 2, (3, 4)]
        ['a', 'b', 'e', ('hello', 'hallo')]
        ['a', 'b', 'f', (4, None)]

        >>> flag, path = diff(a, b, return_str_span=True)
        >>> for p in path: p
        ['a', 'b', 'c', (1.1, 1.2)]
        ['a', 'b', 'd', 2, (3, 4)]
        ['a', 'b', 'e', (((1, 2),), ((1, 2),))]
        ['a', 'b', 'f', (4, None)]

    """
    paths = []
    flags = []
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        keys = set(obj1.keys()) | set(obj2.keys())
        for k in keys:
            o1 = obj1.get(k)
            o2 = obj2.get(k)
            flag, _path = diff(o1, o2, **kwargs)
            flags.append(flag)
            if not flag:
                if _path:
                    for p in _path:
                        paths.append([k] + p)
                else:
                    paths.append([k])

    elif isinstance(obj1, (list, tuple, set)) and isinstance(obj2, (list, tuple)):
        for i, (o1, o2) in enumerate(zip(obj1, obj2)):
            flag, _path = diff(o1, o2, **kwargs)
            flags.append(flag)
            if not flag:
                if _path:
                    for p in _path:
                        paths.append([i] + p)
                else:
                    paths.append([i])
    else:
        flag, path = diff_obj(obj1, obj2, **kwargs)
        flags.append(flag)
        paths.append(path)

    return all(flags), paths
