import time
from functools import wraps
from .os_lib import FakeIo


class IgnoreException:
    """
    Usage:
        .. code-block:: python

            from utils.os_lib import AutoLog
            ignore_exception = IgnoreException()

            class SimpleClass:
                @ignore_exception.add_ignore()
                @ignore_exception.add_ignore(error_message='there is an error')
                @ignore_exception.add_ignore(err_type=Exception)
                def func(self):
                    ...
    """

    def __init__(self, verbose=True, stdout_method=print):
        self.stdout_method = stdout_method if verbose else FakeIo()

    def add_ignore(
            self,
            error_message='',
            err_type=(ConnectionError, TimeoutError)
    ):
        def wrap2(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                try:
                    return func(*args, **kwargs)

                except err_type as e:
                    msg = error_message or f'Something error occur: {e}'
                    self.stdout_method(msg)

            return wrap

        return wrap2


class Retry:
    """
    Usage:
        .. code-block:: python

            from utils.os_lib import Retry

            retry = Retry()

            class SimpleClass:
                @retry.add_try()
                @retry.add_try(error_message='there is an error, sleep %d seconds')
                @retry.add_try(err_type=Exception)
                def func(self):
                    ...
    """

    def __init__(self, verbose=True, stdout_method=print, count=3, wait=15):
        self.verbose = verbose
        self.stdout_method = stdout_method if verbose else FakeIo()
        self.count = count
        self.wait = wait

    def add_try(
            self,
            error_message='',
            err_type=(ConnectionError, TimeoutError)
    ):
        def wrap2(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                for i in range(self.count):
                    try:
                        return func(*args, **kwargs)

                    except err_type as e:
                        if i >= self.count - 1:
                            raise e

                        msg = error_message or f'Something error occur, sleep %d seconds, and then retry'
                        msg = msg % self.wait
                        self.stdout_method(msg)
                        time.sleep(self.wait)
                        self.stdout_method(f'{i + 2}th try!')

            return wrap

        return wrap2


ignore_exception = IgnoreException()
retry = Retry()
