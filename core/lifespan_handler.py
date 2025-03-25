#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @文件       :lifespan_handler.py
# @时间       :2023/12/5 下午4:53
# @作者       :lihb
# @说明       :
import concurrent.futures
import os
from contextlib import asynccontextmanager

import jieba
from fastapi import FastAPI

from utils.m3e import GetM3eModel


# from utils.images_vector.model_factory import OpenAIClip, GoogleClip


@asynccontextmanager
async def lifespan(_: FastAPI):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(GetM3eModel.start_model)
        # executor.submit(OpenAIClip.initialize_model)
        # executor.submit(GoogleClip.initialize_model)
    # OpenAIClip.initialize_model()
    # GoogleClip.initialize_model()
    if os.name != 'nt':
        jieba.enable_parallel(4)
    jieba.initialize()  # 手动初始化（可选）
    yield
    if os.name != 'nt':
        jieba.disable_parallel()
