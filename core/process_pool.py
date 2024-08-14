#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
from concurrent.futures import ProcessPoolExecutor

from core.log import setup_logging


class ProcessPool:
    executor: ProcessPoolExecutor = None

    @classmethod
    def get_executor(cls) -> ProcessPoolExecutor:
        return cls.executor

    @classmethod
    def start_executor(cls) -> ProcessPoolExecutor:
        cls.executor = ProcessPoolExecutor()
        return cls.executor

    @classmethod
    def stop_executor(cls) -> None:
        cls.executor.shutdown()
