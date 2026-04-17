import logging
import warnings

from utils import log_utils, web_app, converter, op_utils
from typing import Optional, Annotated
from functools import partial

LOGGING = 'logging'
LOGURU = 'loguru'
WANDB = 'wandb'
TENSORBOARD = 'tensorboard'

make_logger_fn = op_utils.RegisterTables()
empty_logger = log_utils.EmptyLogger()


class LogHooks:
    """
    .. code-block:: python

        process = LogHooks()

        # add a logger module, e.g., add loguru logger
        from loguru import logger
        process.default_logger_types = [LOGURU, WANDB]
        process.default_main_logger_type = LOGURU

        # only add a logger func, e.g., add tqdm pbar logger
        from tqdm import tqdm
        pbar = tqdm(iter)
        process.register_logger('pbar', pbar.set_postfix)

        # add a customized logger module
        LOGGER = 'xxx'
        @make_logger_fn.add_register(LOGGER)
        class Logger:
            def int(self, process: LogHooks, looger_name: str):
                raise NotImplementedError

        Logger().init(process, LOGGER)

        process.init()
    """
    default_logger_types = [LOGGING, WANDB]
    default_main_logger_type = LOGGING

    model_version: Annotated[
        str,
        'for work_dir and cache_dir'
    ] = ''
    dataset_version: Annotated[
        str,
        'for work_dir and cache_dir'
    ]
    work_dir: str

    def __init__(self):
        super().__init__()
        self.loggers = set()
        self.trace_log_items = dict()
        self.log_methods = dict()

    def init_logs(self):
        for logger_type in self.default_logger_types:
            logger_cls = make_logger_fn.get(logger_type)
            logger_cls().init(self, logger_type)

    def register_logger(self, name, log_method):
        self.loggers.add(name)
        self.trace_log_items[name] = {}
        self.log_methods[name] = log_method

    def trace(self, item: dict, loggers=None):
        """only cache the log items to `trace_log_items`, and output when calling `log_trace()`"""
        loggers = loggers or [self.default_main_logger_type]
        if not isinstance(loggers, (list, tuple, set)):
            loggers = [loggers]

        for logger in loggers:
            self.trace_log_items[logger].update(item)

    def get_log_trace(self, logger=None):
        """get the items is cached before"""
        logger = logger or self.default_main_logger_type
        return self.trace_log_items[logger]

    def log_trace(self, loggers=None, **kwargs):
        """output the log items which is cached before"""
        loggers = loggers or [self.default_main_logger_type]
        if not isinstance(loggers, (list, tuple, set)):
            loggers = [loggers]

        for logger in loggers:
            item = self.trace_log_items.get(logger)
            self.log_methods.get(logger)(item, **kwargs)
            self.trace_log_items[logger] = {}

    def log(self, item, loggers=None, **kwargs):
        loggers = loggers or [self.default_main_logger_type]
        if not isinstance(loggers, (list, tuple, set)):
            loggers = [loggers]

        for logger in loggers:
            self.log_methods.get(logger, empty_logger)(item, **kwargs)


@make_logger_fn.add_register(LOGGING)
class Logging:
    def init(self, process: LogHooks, looger_name: str):
        log_utils.logger_init(getattr(process, 'log_dir', None))
        logger = log_utils.get_logger(getattr(process, 'logger', None))
        process.register_logger(looger_name, lambda item, level=logging.INFO, **kwargs: logger.log(level, item, stacklevel=3))


@make_logger_fn.add_register(LOGURU)
class Loguru:
    def init(self, process: LogHooks, looger_name: str):
        from loguru import logger
        process.register_logger(looger_name, lambda item, level='INFO', **kwargs: logger.opt(depth=2).log(level, item))


@make_logger_fn.add_register(WANDB)
class Wandb:
    def init(self, process: LogHooks, looger_name: str):
        def _wandb_init(*args, **kwargs):
            # only init wandb runner before training
            process.wandb.login(api_key=getattr(process, 'wandb_api_key', ''))
            wandb_run = process.wandb.init(
                project=process.model_version,
                name=process.dataset_version,
                dir=f'{process.work_dir}',
                id=getattr(process, 'wandb_id', None),
                resume=True
            )
            # for retraining
            process.wandb_id = wandb_run.id

        if getattr(process, 'use_wandb', False):
            wandb = self.make_wandb()
        else:
            wandb = log_utils.FakeWandb()

        process.wandb = wandb
        process.register_logger(looger_name, lambda item, **kwargs: wandb.log(item))
        process.register_train_start(_wandb_init)
        process.register_train_end(wandb.finish)

    def make_wandb(self):
        try:
            # note, can replace wandb to swanlab now
            # import swanlab as wandb
            import wandb
        except ImportError:
            wandb = log_utils.FakeWandb()
            warnings.warn('Wandb import error, wandb init fail, please check install')

        return wandb


@make_logger_fn.add_register(TENSORBOARD)
class Tensorboard:
    def int(self, process: LogHooks, looger_name: str):
        raise NotImplementedError


class ApiHooks:
    def __init__(self):
        super().__init__()

        self.api_funcs = ['single_predict', 'batch_predict']
        self.api_config = {
            '/': {k: {
                'func': partial(self.api_func_wrap, func=getattr(self, k), name=k)
            } for k in self.api_funcs
            }
        }

        self.web_app_op = web_app.FlaskOp

    def create_app(self):
        return self.web_app_op.from_configs(self.api_config)

    def api_func_wrap(self, data, func, name):
        ret = func(*data.get('data', []), **data.get('params', {}))
        return converter.DataConvert.custom_to_constant(ret)
