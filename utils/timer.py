#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
import threading
import time
from functools import wraps

from loguru import logger


# from devops_api.libs.utils import logger


def timer(function):
    """装饰器函数timer

    :param function:想要计时的函数
    :return:
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        def_name = function.__name__

        time_start = time.time()
        thread_name = threading.current_thread().name
        res = function(*args, **kwargs)
        cost_time = time.time() - time_start
        logger.info(f'{def_name},线程:{thread_name},运行时间：{cost_time}秒')
        # print(f'{def_name},线程:{thread_name},运行时间：{cost_time}秒')
        return res

    return wrapper


class Timer:
    def __init__(self, msg: str | int = None):
        """ Timer类用于测量函数执行时间和记录日志。

        :param msg: 日志信息的描述
        :type msg: str or int
        """
        self.msg = msg

    def __call__(self, func):
        """ __call__方法使Timer类的实例可以像函数一样被调用。

        :param func: 被装饰的函数
        :type func: function
        :return: 装饰后的函数
        """
        # if asyncio.iscoroutinefunction(func):
        #     return self._async_wrapper(func)
        # else:
        return self._wrapper(func)

    def _wrapper(self, func):
        """ 同步函数的装饰器。

        :param func: 被装饰的同步函数
        :type func: function
        :return: 装饰后的同步函数
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            time_start = time.time()
            res = func(*args, **kwargs)
            self._log_execution_time(func, time_start)
            return res

        return wrapper

    def _log_execution_time(self, func, start_time):
        """ 记录函数执行时间的日志。

        :param func: 函数对象
        :type func: function
        :param start_time: 函数开始执行的时间戳
        :type start_time: float
        """
        end_time = time.time()
        execution_time = end_time - start_time
        module_name = func.__module__
        function_name = func.__name__
        message = f"{self.msg}, " if self.msg else ""
        log_message = f"{message} 运行时间：{execution_time:.5f}秒"
        logger.patch(
                lambda r: r.update(function=f"{r.get('function')} ({module_name}:{function_name})")

        ).info(log_message)


class AsyncTimer(Timer):
    def __call__(self, func):
        return self._async_wrapper(func)

    def _async_wrapper(self, func):
        """ 异步函数的装饰器。

        :param func: 被装饰的异步函数
        :type func: function
        :return: 装饰后的异步函数
        """

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            time_start = time.time()
            res = await func(*args, **kwargs)
            await self._log_execution_time(func, time_start)
            return res

        return async_wrapper

    async def _log_execution_time(self, func, start_time):
        super()._log_execution_time(func, start_time)
