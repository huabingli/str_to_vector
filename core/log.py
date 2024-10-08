#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @文件       :log.py
# @时间       :2023/9/21 上午11:06
# @作者       :lihb
# @说明       :
import inspect
import logging
import sys
import threading

from loguru import logger

from core.config import request_id_var, request_time_it_var, settings


def _logger_filter(record):
    record['extra'].update(request_id_var=request_id_var.get(None))
    record['extra'].update(request_time_it_var=request_time_it_var.get('').strip())
    return record


class InterceptHandler(logging.Handler):

    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # print(level)
        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


lock = threading.Lock()


def stop_logging() -> None:
    models = ['httpcore', 'httpx', 'apscheduler', 'elastic_transport', 'sqlalchemy', 'watchfiles']
    for model in models:
        logger.disable(model)


def setup_logging():
    # LOG_LEVEL = logging.getLevelName(settings.log_level)

    # intercept everything at the root logger
    # logging.root.handlers = [InterceptHandler()]
    # logging.root.setLevel(LOG_LEVEL)
    with lock:
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

        # remove every other logger's handlers
        # and propagate to root logger
        for name in logging.root.manager.loggerDict.keys():
            logging.getLogger(name).handlers = []
            logging.getLogger(name).propagate = True
        logger.remove()  # Will remove all handlers already configured
        # configure loguru
        logger.configure(patcher=_logger_filter)
        stop_logging()
        logger.add(sys.stdout, level=settings.log_level, colorize=settings.log_colorize,
                   format='[<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>] '
                          '[<magenta>{process.name}</magenta>:<yellow>{thread.name}</yellow>] '
                          '[<cyan>{name}</cyan>:<cyan>{function}</cyan>:<yellow>{line}</yellow>] '
                          '[<level>{level}</level>] '
                          '[{extra[request_id_var]}] '
                          '[{extra[request_time_it_var]}] '
                          '<level>{message}</level>',
                   )
        # logger.level(settings.log_level)
        logger.info(f'初始化日志配置 {settings.log_level}')
