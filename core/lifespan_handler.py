#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @文件       :lifespan_handler.py
# @时间       :2023/12/5 下午4:53
# @作者       :lihb
# @说明       :
from contextlib import asynccontextmanager

from fastapi import FastAPI

from core.config import settings
from utils.images_vector.model_factory import GoogleClip, OpenAIClip


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.default_provider == 'google':
        GoogleClip.initialize_model()
    elif settings.default_provider == 'openai':
        OpenAIClip.initialize_model()
    yield
    await GoogleClip.close_httpx()
    await OpenAIClip.close_httpx()
