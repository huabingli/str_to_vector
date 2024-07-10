#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @文件       :lifespan_handler.py
# @时间       :2023/12/5 下午4:53
# @作者       :lihb
# @说明       :
from contextlib import asynccontextmanager

from fastapi import FastAPI

from utils.m3e import GetM3eModel


@asynccontextmanager
async def lifespan(_: FastAPI):
    GetM3eModel.start_model()
    yield
