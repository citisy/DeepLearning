import os
import time
import psutil
import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
from functools import wraps
from . import os_lib, converter, configs


class EmptyLogger:
    def __call__(self, *args, **kwargs):
        pass

    def debug(self, msg, *args, **kwargs):
        pass

    def info(self, msg, *args, **kwargs):
        pass

    def warning(self, msg, *args, **kwargs):
        pass

    def error(self, msg, *args, **kwargs):
        pass

    def critical(self, msg, *args, **kwargs):
        pass


class FakeLogger:
    def debug(self, msg, *args, **kwargs):
        print(msg)

    def info(self, msg, *args, **kwargs):
        print(msg)

    def warning(self, msg, *args, **kwargs):
        print(msg)

    def error(self, msg, *args, **kwargs):
        print(msg)

    def critical(self, msg, *args, **kwargs):
        print(msg)


class FakeWandb:
    def __init__(self, *args, **kwargs):
        self.id = None
        self.__dict__.update(**kwargs)

    def init(self, *args, **kwargs):
        return self

    def Table(self, *args, **kwargs):
        return self

    def Image(self, *args, **kwargs):
        return self

    def log(self, *args, **kwargs):
        pass

    def finish(self, *args, **kwargs):
        pass


class MultiProcessTimedRotatingFileHandler(TimedRotatingFileHandler):
    @property
    def dfn(self):
        current_time = int(time.time())
        # get the time that this sequence started at and make it a TimeTuple
        dst_now = time.localtime(current_time)[-1]
        t = self.rolloverAt - self.interval
        if self.utc:
            time_tuple = time.gmtime(t)
        else:
            time_tuple = time.localtime(t)
            dst_then = time_tuple[-1]
            if dst_now != dst_then:
                if dst_now:
                    addend = 3600
                else:
                    addend = -3600
                time_tuple = time.localtime(t + addend)
        dfn = self.rotation_filename(self.baseFilename + "." + time.strftime(self.suffix, time_tuple))

        return dfn

    def shouldRollover(self, record):
        """
        是否应该执行日志滚动操作：
        1、存档文件已存在时，执行滚动操作
        2、当前时间 >= 滚动时间点时，执行滚动操作
        """
        dfn = self.dfn
        t = int(time.time())
        if t >= self.rolloverAt or os.path.exists(dfn):
            return 1
        return 0

    def doRollover(self):
        """
        执行滚动操作
        1、文件句柄更新
        2、存在文件处理
        3、备份数处理
        4、下次滚动时间点更新
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        # get the time that this sequence started at and make it a TimeTuple

        dfn = self.dfn

        # 存档log 已存在处理
        if not os.path.exists(dfn):
            self.rotate(self.baseFilename, dfn)

        # 备份数控制
        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)

        # 延迟处理
        if not self.delay:
            self.stream = self._open()

        # 更新滚动时间点
        current_time = int(time.time())
        new_rollover_at = self.computeRollover(current_time)
        while new_rollover_at <= current_time:
            new_rollover_at = new_rollover_at + self.interval

        # If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dst_at_rollover = time.localtime(new_rollover_at)[-1]
            dst_now = time.localtime(current_time)[-1]
            if dst_now != dst_at_rollover:
                if not dst_now:  # DST kicks in before next rollover, so we need to deduct an hour
                    addend = -3600
                else:  # DST bows out before next rollover, so we need to add an hour
                    addend = 3600
                new_rollover_at += addend
        self.rolloverAt = new_rollover_at


def logger_init(log_dir=None, **custom_config):
    """logging配置
    默认loggers：['', 'basic', 'service_standard', 'service', '__main__']

    Usage:
        .. code-block:: python

            import logging
            from utils.configs import logger_init

            # default init
            logger_init()

            # log print to file
            logger_init('logs')

            # add custom config
            logger_init(handlers={...}, loggers={...})

            logger = logging.getLogger('service')
            logger.info('')

    """

    default_config = {
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
        # handlers to scream
        'handlers': {
            # 屏幕输出流
            'default': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stderr',
            },

            # 简单的无格式屏幕输出流
            'print': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stderr',
            },
        },
        'loggers': {
            # root logger
            '': {
                'handlers': ['default'],
                'level': 'INFO',
                'propagate': False
            },

            # 简单的无格式屏幕输出流
            'print': {
                'handlers': ['print'],
                'level': 'INFO',
                'propagate': False
            }
        }
    }

    if log_dir is not None:  # add file handles
        os_lib.mk_dir(log_dir)
        add_config = {
            # handlers to file
            'handlers': {
                # 简略信息info
                'info_standard': {
                    'level': 'INFO',
                    'formatter': 'standard',
                    'class': 'utils.log_utils.MultiProcessTimedRotatingFileHandler',
                    'filename': f'{log_dir}/info_standard.log',
                    'when': 'W0',
                    'backupCount': 5,
                },

                # 详细信息info
                'info': {
                    'level': 'INFO',
                    'formatter': 'precise',
                    'class': 'utils.log_utils.MultiProcessTimedRotatingFileHandler',
                    'filename': f'{log_dir}/info.log',
                    'when': 'D',
                    'backupCount': 15,
                },

                # 详细信息error
                'error': {
                    'level': 'ERROR',
                    'formatter': 'precise',
                    'class': 'utils.log_utils.MultiProcessTimedRotatingFileHandler',
                    'filename': f'{log_dir}/error.log',
                    'when': 'W0',
                    'backupCount': 5,
                },
            },

            'loggers': {
                # root logger
                '': {
                    'handlers': ['default', 'info_standard', 'error'],
                    'level': 'INFO',
                    'propagate': False
                },

                # 简单的无格式屏幕输出流
                'print': {
                    'handlers': ['print', 'info_standard', 'error'],
                    'level': 'INFO',
                    'propagate': False
                },

                'service': {
                    'handlers': ['default', 'info', 'error'],
                    'level': 'INFO',
                    'propagate': False
                },
            }

        }
        default_config = configs.merge_dict(default_config, add_config)

    default_config = configs.merge_dict(default_config, custom_config)
    logging.config.dictConfig(default_config)
    return default_config


def wandb_init(**custom_config):
    import wandb
    default_config = {

    }
    default_config = configs.merge_dict(default_config, custom_config)

    wandb.init(project='test')
    return default_config


def get_logger(logger=''):
    if isinstance(logger, str) or logger is None:
        logger = logging.getLogger(logger)
    return logger


class AutoLog:
    """
    Usage:
        .. code-block:: python

            from utils.os_lib import AutoLog
            auto_log = AutoLog()

            class SimpleClass:
                @auto_log.memory_log('success')
                @auto_log.memory_log()
                @auto_log.time_log()
                def func(self):
                    ...
    """

    def __init__(self, verbose=True, stdout_method=print, is_simple_log=True, is_time_log=True, is_memory_log=True):
        self.stdout_method = stdout_method if verbose else os_lib.FakeIo()
        self.is_simple_log = is_simple_log
        self.is_time_log = is_time_log
        self.is_memory_log = is_memory_log

    def simple_log(self, string):
        def wrap2(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                r = func(*args, **kwargs)
                if self.is_simple_log:
                    self.stdout_method(string)
                return r

            return wrap

        return wrap2

    def time_log(self, prefix_string=''):
        def wrap2(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                if self.is_time_log:
                    st = time.time()
                    r = func(*args, **kwargs)
                    et = time.time()
                    self.stdout_method(f'{prefix_string} - elapse[{et - st:.3f}s]!')
                else:
                    r = func(*args, **kwargs)
                return r

            return wrap

        return wrap2

    def memory_log(self, prefix_string=''):
        def wrap2(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                if self.is_memory_log:
                    a = MemoryInfo.get_process_mem_info()
                    r = func(*args, **kwargs)
                    b = MemoryInfo.get_process_mem_info()
                    self.stdout_method(f'{prefix_string}\nbefore: {a}\nafter: {b}')
                else:
                    r = func(*args, **kwargs)
                return r

            return wrap

        return wrap2


class MemoryInfo:
    from .visualize import TextVisualize

    @classmethod
    def get_process_mem_info(cls, pretty_output=True):
        """
        uss, 进程独立占用的物理内存（不包含共享库占用的内存）
        rss, 该进程实际使用物理内存（包含共享库占用的全部内存）
        vms, 虚拟内存总量
        """
        pid = os.getpid()
        p = psutil.Process(pid)
        info = p.memory_full_info()
        info = {
            'pid': str(pid),
            'uss': info.uss,
            'rss': info.rss,
            'vms': info.vms,
        }

        if pretty_output:
            for k, v in info.items():
                if k != 'pid':
                    info[k] = cls.TextVisualize.num_to_human_readable_str(v)

        return info

    @classmethod
    def get_cpu_mem_info(cls, pretty_output=True):
        """
        percent, 实际已经使用的内存占比
        total, 内存总的大小
        available, 还可以使用的内存
        free, 剩余的内存
        used, 已经使用的内存
        """
        info = dict(psutil.virtual_memory()._asdict())
        if pretty_output:
            for k, v in info.items():
                if k != 'percent':
                    info[k] = cls.TextVisualize.num_to_human_readable_str(v)

        return info

    @classmethod
    def get_gpu_mem_info(cls, device=0, pretty_output=True):
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        info = dict(
            total=info.free,
            used=info.used,
            free=info.free,
        )
        if pretty_output:
            for k, v in info.items():
                info[k] = cls.TextVisualize.num_to_human_readable_str(v)

        return info

    @classmethod
    def get_mem_info(cls, pretty_output=True):
        info = dict(
            process_mem=cls.get_process_mem_info(pretty_output),
            env_mem=cls.get_cpu_mem_info(pretty_output),
            gpu_mem=cls.get_gpu_mem_info(pretty_output),
        )

        return info


def class_info(ins):
    import inspect
    args = inspect.getfullargspec(ins.__init__).args
    args.pop(0)
    return dict(
        doc=ins.__doc__,
        args=args
    )


def get_class_annotations(cls):
    if not hasattr(cls, '__annotations__'):
        return dict()

    annotations = cls.__annotations__

    for parent_cls in cls.__bases__:
        parent_annotations = get_class_annotations(parent_cls)
        annotations.update(parent_annotations)

    return annotations
