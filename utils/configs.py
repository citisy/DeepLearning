import yaml
import configparser
import logging
import copy
from . import os_lib


def load_config_from_yml(path) -> dict:
    return yaml.load(open(path, 'rb'), Loader=yaml.Loader)


def load_config_from_ini(path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")
    return config


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
            new_dic.update(cur_str(k, v, new_dic))

        return new_dic

    return cur_dict(d, {})


def merge_dict(d1: dict, d2: dict):
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


def logger_init(config={}, log_dir='logs'):
    """logging配置
    默认loggers：['', 'basic', 'service_standard', 'service', '__main__']"""

    default_logging_config = {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'standard': {
                'format': '[ %(asctime)s ] [%(levelname)s] [%(name)s]: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'precise': {
                'format': '[ %(asctime)s ] [%(levelname)s] [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            # 屏幕输出流
            'default': {
                'level': 'DEBUG',
                'formatter': 'precise',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stdout',
            },

            # 文件输出流（简略info）
            'info_standard': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'filename': f'{log_dir}/info_standard.log',
                'when': 'W0',
                'backupCount': 5,
            },

            # 文件输出流（详细info级别以上）
            'info': {
                'level': 'INFO',
                'formatter': 'precise',
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'filename': f'{log_dir}/info.log',
                'when': 'D',
                'backupCount': 15,
            },

            # 文件输出流（详细error级别以上）
            'error': {
                'level': 'ERROR',
                'formatter': 'precise',
                'class': 'logging.handlers.TimedRotatingFileHandler',
                'filename': f'{log_dir}/error.log',
                'when': 'W0',
                'backupCount': 5,
            },

            # 文件输出流（详细critical级别以上）
            'critical': {
                'level': 'CRITICAL',
                'formatter': 'precise',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': f'{log_dir}/critical.log',
            },

        },
        'loggers': {
            # root logger
            '': {
                'handlers': ['default'],
                'level': 'WARNING',
                'propagate': False
            },
            'basic': {
                'handlers': ['default', 'info_standard', 'error', 'critical'],
                'level': 'INFO',
                'propagate': False
            },
            'service_standard': {
                'handlers': ['default', 'info_standard', 'error', 'critical'],
                'level': 'INFO',
                'propagate': False
            },
            'service': {
                'handlers': ['default', 'info', 'error', 'critical'],
                'level': 'INFO',
                'propagate': False
            },
            # if __name__ == '__main__'
            '__main__': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': False
            },
        }
    }

    default_logging_config = merge_dict(default_logging_config, config)

    os_lib.mk_dir(log_dir)
    logging.config.dictConfig(default_logging_config)
