import copy
from . import os_lib, converter


class ArgDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax.
    so that it can be treated as `argparse.ArgumentParser().parse_args()`"""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


def collapse_dict(d: dict):
    """

    Example:
        >>> d = {'a': {'b': 1, 'c': 2, 'e': {'f': 4}}, 'd': 3}
        >>> collapse_dict(d)
        >>> {'a.b': 1, 'a.c': 2, 'a.e.f': 4, 'd': 3}

    """

    def cur(cur_dic, cur_k, new_dic):
        for k, v in cur_dic.items():
            if isinstance(v, dict):
                k = f'{cur_k}.{k}'
                cur(v, k, new_dic)
            else:
                new_dic[f'{cur_k}.{k}'] = v

        return new_dic

    new_dic = cur(d, '', {})
    new_dic = {k[1:]: v for k, v in new_dic.items()}
    return new_dic


def expand_dict(d: dict):
    """expand dict while '.' in key or '=' in value

    Example:
        >>> d = {'a.b': 1}
        >>> expand_dict(d)
        {'a': {'b': 1}}

        >>> d = {'a': 'b=1'}
        >>> expand_dict(d)
        {'a': {'b': 1}}

        >>> d = {'a.b.c.d': 1, 'a.b': 'c.e=2', 'a.b.e': 3}
        >>> expand_dict(d)
        {'a': {'b': {'c': {'d': 1, 'e': '2'}, 'e': 3}}}
    """

    def cur_str(k, v, cur_dic):
        if '.' in k:
            a, b = k.split('.', 1)
            v = cur_str(b, v, cur_dic.get(a, {}))
            return {a: v}
        elif isinstance(v, dict):
            cur_dic[k] = cur_dict(v, cur_dic.get(k, {}))
            return cur_dic
        else:
            if isinstance(v, str) and '=' in v:
                kk, vv = v.split('=', 1)
                v = cur_dict({kk.strip(): vv.strip()}, cur_dic.get(k, {}))
            cur_dic[k] = v
            return cur_dic

    def cur_dict(cur_dic, new_dic):
        for k, v in cur_dic.items():
            new_dic = merge_dict(new_dic, cur_str(k, v, new_dic))

        return new_dic

    return cur_dict(d, {})


def merge_dict(d1: dict, d2: dict) -> dict:
    """merge values from d1 and d2
    if had same key, d2 will cover d1

    Example:
        >>> d1 = {'a': {'b': {'c': 1}}}
        >>> d2 = {'a': {'b': {'d': 2}}}
        >>> merge_dict(d1, d2)
        {'a': {'b': {'c': 1, 'd': 2}}}

    """

    def cur(cur_dic, new_dic):
        for k, v in new_dic.items():
            if k in cur_dic and isinstance(v, dict) and isinstance(cur_dic[k], dict):
                v = cur(cur_dic[k], v)

            cur_dic[k] = v

        return cur_dic

    return cur(copy.deepcopy(d1), copy.deepcopy(d2))


def permute_obj(obj: dict or list):
    """

    Example:
        
        >>> kwargs = [{'a': [1], 'b': [2, 3]}, {'c': [4, 5, 6]}]
        >>> permute_obj(kwargs)
        [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}, {'c': 4}, {'c': 5}, {'c': 6}]

    """

    def cur(cur_obj: dict):
        r = [{}]
        for k, v in cur_obj.items():
            r = [{**rr, k: vv} for rr in r for vv in v]

        return r

    ret = []
    if isinstance(obj, dict):
        ret += cur(obj)
    else:
        for o in obj:
            ret += cur(o)

    return ret


def default(*args):
    """check the items by order and return the first item which is not None"""
    for obj in args:
        if obj is not None:
            return obj


def parse_params_example(path, parser) -> dict:
    """an example for parse parameters"""

    def params_params_from_file(path) -> dict:
        """user params, low priority"""
        return expand_dict(os_lib.loader.load_yaml(path))

    def params_params_from_env(flag='Global.') -> dict:
        """global params, middle priority"""
        import os

        args = {}
        for k, v in os.environ.items():
            if k.startswith(flag):
                k = k.replace(flag, '')
                args[k] = v

        config = expand_dict(args)
        config = converter.DataConvert.complex_str_to_constant(config)

        return config

    def params_params_from_arg(parser) -> dict:
        """local params, high priority
        # parser will be created like that
        import argparse

        parser = argparse.ArgumentParser()
        ...
        parser.add_argument('-c', '--config', nargs='+', default=[], help='global config')
        """

        args = parser.parse_args()
        _config = args.config
        if _config:
            _config = dict(s.split('=') for s in _config)
            _config = expand_dict(_config)
            _config = converter.DataConvert.complex_str_to_constant(_config)
        else:
            _config = {}

        return _config

    config = params_params_from_file(path)
    config = merge_dict(config, params_params_from_env())
    config = merge_dict(config, params_params_from_arg(parser))

    return config


def parse_pydantic(model: 'pydantic.BaseModel', return_example=False) -> dict:
    schema = model.schema()
    ret = parse_pydantic_schema(schema, schema.get('definitions', {}))
    if return_example:
        ret = parse_pydantic_dict(ret)

    return ret


def parse_pydantic_schema(schema: dict, definitions={}) -> dict:
    ret = {}
    required = schema.get('required', [])
    for name, attr in schema['properties'].items():
        types = parse_pydantic_attr(attr)
        for i, _type in enumerate(types):
            if isinstance(_type, dict):
                for v in _type.values():
                    for ii, __type in enumerate(v['types']):
                        if __type in definitions:
                            v['types'][ii] = parse_pydantic_schema(definitions[__type], definitions)
            else:
                if _type in definitions:
                    types[i] = parse_pydantic_schema(definitions[_type], definitions)

        ret[name] = dict(
            types=types,
            is_required=name in required,
        )

    return ret


def parse_pydantic_attr(attr: dict) -> list:
    def parse(a):
        if 'type' in a:
            _type = a['type']
        elif '$ref' in a:
            obj = a['$ref']
            _type = obj.split('/')[-1]
        else:
            _type = ''
        return _type

    types = []
    tmp = types
    a = attr
    while 'items' in a or 'additionalProperties' in a or 'allOf' in a:
        if 'items' in a:
            _type = parse(a)
            tmp.append(_type)
            a = a['items']

        elif 'additionalProperties' in a:
            _type = parse(a)
            _tmp = []
            tmp.append({_type: dict(types=_tmp, is_required=True)})
            tmp = _tmp
            a = a['additionalProperties']

        elif 'allOf' in a:
            _type = parse(a['allOf'][0])
            a = a['allOf'][0]

    _type = parse(a)
    if _type:
        tmp.append(_type)

    return types


def parse_pydantic_dict(ret: dict) -> dict:
    d = {}
    for k, v in ret.items():
        a = []
        b = a
        for _type in v['types']:
            if isinstance(_type, dict):
                b.append(parse_pydantic_dict(_type))
            elif _type == 'array':
                c = []
                b.append(c)
                b = c
            else:
                b.append(_type)
        d[k] = a[0]

    return d
