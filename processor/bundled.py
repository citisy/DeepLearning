import logging
from utils import log_utils
from typing import Optional

BASELOG = 1
WANDB = 2
TENSORBOARD = 3

empty_logger = log_utils.EmptyLogger()


class LogHooks:
    def __init__(self):
        super().__init__()
        self.loggers = [BASELOG, WANDB]
        self._init_trace_log_items()
        self.log_methods = {
            BASELOG: self.log_base,
            WANDB: self.log_wandb
        }

    def register_logger(self, name, log_method):
        self.loggers.append(name)
        self.log_methods[name] = log_method
        self._init_trace_log_items()

    def _init_trace_log_items(self):
        self.trace_log_items = {logger: {} for logger in self.loggers}

    logger: Optional
    wandb: Optional

    def init_log_base(self, log_dir):
        log_utils.logger_init(log_dir)
        self.logger = log_utils.get_logger()

    use_wandb = False

    def init_wandb(self):
        if self.use_wandb:
            try:
                import wandb
            except ImportError:
                wandb = log_utils.FakeWandb()
                self.logger.warning('wandb import error, wandb init fail, please check install')

        else:
            from utils.os_lib import FakeWandb
            wandb = FakeWandb()

        self.wandb = wandb

    def trace(self, item: dict, loggers=BASELOG):
        if isinstance(loggers, int):
            loggers = [loggers]

        for logger in loggers:
            self.trace_log_items[logger].update(item)

    def get_log_trace(self, logger=BASELOG):
        return self.trace_log_items[logger]

    def log_trace(self, loggers=BASELOG, **kwargs):
        if isinstance(loggers, int):
            loggers = [loggers]

        for logger in loggers:
            item = self.trace_log_items.get(logger)
            self.log_methods.get(logger)(item, **kwargs)
            self.trace_log_items[logger] = {}

    def log(self, item, loggers=BASELOG, **kwargs):
        if not isinstance(loggers, (list, tuple)):
            loggers = [loggers]

        for logger in loggers:
            self.log_methods.get(logger, empty_logger)(item, **kwargs)

    def log_base(self, item, level=logging.INFO, **kwargs):
        self.logger.log(level, item)

    def log_wandb(self, item, **kwargs):
        self.wandb.log(item)
