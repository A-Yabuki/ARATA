# coding: utf-8

import time
from functools import singledispatch, update_wrapper, wraps

from . import error_handler
from .enum import ErrorType

def timeit(func):

    @wraps
    def wrapper(*args, **kwargs):

        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()

        consumed_time = end_time - start_time

        return consumed_time

    return wrapper


def watch_error(error_msg: str, error_type: ErrorType):

    def decorate(func):
    
        def wrapper(*args, **kwargs):
            try:
                res = func(*args, **kwargs)
                return res
            except:
                error_handler.ChainedErrorHandler(error_msg, error_type)
        return wrapper

    return decorate


def methoddispatch(func):

    dispatcher = singledispatch(func)
    
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)

    return wrapper
