import logging
from utils import log_utils
from typing import Optional

LOGGING = 'logging'
WANDB = 'wandb'
TENSORBOARD = 'tensorboard'

empty_logger = log_utils.EmptyLogger()


class LogHooks:
    """
    .. code-block:: python

        process = LogHooks()

        # add tqdm pbar logger
        from tqdm import tqdm
        pbar = tqdm(iter)
        process.register_logger('pbar', pbar.set_postfix)

        # add loguru logger
        from loguru import logger
        process.register_logger('loguru', lambda item, level='INFO', **kwargs: logger.opt(depth=2).log(level, item))

    """

    def __init__(self):
        super().__init__()
        self.loggers = set()
        self.trace_log_items = dict()
        self.log_methods = dict()

    def register_logger(self, name, log_method):
        self.loggers.add(name)
        self.trace_log_items[name] = {}
        self.log_methods[name] = log_method

    logger: Optional

    def init_log_base(self, log_dir=None, logger=None):
        log_utils.logger_init(log_dir)
        logger = log_utils.get_logger(logger)
        self.register_logger(LOGGING, lambda item, level=logging.INFO, **kwargs: logger.log(level, item, stacklevel=3))

    use_wandb = False
    wandb: Optional

    model_version: str
    dataset_version: str
    work_dir: str
    wandb_id: str
    register_train_start: 'model_process.ModelHooks.register_train_start'
    register_train_end: 'model_process.ModelHooks.register_train_end'

    def init_wandb(self):
        if self.use_wandb:
            try:
                import wandb
            except ImportError:
                wandb = log_utils.FakeWandb()
                self.log('wandb import error, wandb init fail, please check install', level=logging.WARNING)

        else:
            wandb = log_utils.FakeWandb()

        self.wandb = wandb
        self.register_logger(WANDB, lambda item, **kwargs: wandb.log(item))
        self.register_train_start(self._wandb_init)
        self.register_train_end(wandb.finish)

    def _wandb_init(self):
        # only init wandb runner before training
        wandb_run = self.wandb.init(
            project=self.model_version,
            name=self.dataset_version,
            dir=f'{self.work_dir}',
            id=self.__dict__.get('wandb_id'),
            reinit=True
        )
        # for retraining
        self.wandb_id = wandb_run.id

    def trace(self, item: dict, loggers=LOGGING):
        """only cache the log items to `trace_log_items`, and output when calling `log_trace()`"""
        if not isinstance(loggers, (list, tuple, set)):
            loggers = [loggers]

        for logger in loggers:
            self.trace_log_items[logger].update(item)

    def get_log_trace(self, logger=LOGGING):
        """get the items is cached before"""
        return self.trace_log_items[logger]

    def log_trace(self, loggers=LOGGING, **kwargs):
        """output the log items which is cached before"""
        if not isinstance(loggers, (list, tuple, set)):
            loggers = [loggers]

        for logger in loggers:
            item = self.trace_log_items.get(logger)
            self.log_methods.get(logger)(item, **kwargs)
            self.trace_log_items[logger] = {}

    def log(self, item, loggers=LOGGING, **kwargs):
        if not isinstance(loggers, (list, tuple, set)):
            loggers = [loggers]

        for logger in loggers:
            self.log_methods.get(logger, empty_logger)(item, **kwargs)


class ApiHooks:
    pass
