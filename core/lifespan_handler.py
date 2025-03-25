#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @文件       :lifespan_handler.py
# @时间       :2023/12/5 下午4:53
# @作者       :lihb
# @说明       :
from contextlib import asynccontextmanager

from fastapi import FastAPI

from utils.images_vector.model_factory import GoogleClip


@asynccontextmanager
async def lifespan(_: FastAPI):
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     executor.submit(GetM3eModel.start_model),
    #     executor.submit(OpenAIClip.initialize_model),
    #     executor.submit(GoogleClip.initialize_model)
    #
    #     for future in concurrent.futures.as_completed(futures):
    #         try:
    #             result = future.result()  # 获取任务结果，若有异常会在这里抛出
    #             print("模型启动成功:", result)
    #         except Exception as e:
    #             print(f"模型初始化失败: {e}")
    # OpenAIClip.initialize_model()
    # GoogleClip.initialize_model()
    # GetM3eModel.start_model()
    # OpenAIClip.initialize_model()
    GoogleClip.initialize_model()
    # if os.name != 'nt':
    #     jieba.enable_parallel(4)
    # jieba.initialize()  # 手动初始化（可选）
    yield
    # if os.name != 'nt':
    #     jieba.disable_parallel()
