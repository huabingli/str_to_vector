#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @文件       :middleware.py
# @时间       :2023/9/21 上午11:15
# @作者       :lihb
# @说明       : 中间件
import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from starlette.requests import Request

from core.config import request_id_var, request_time_it_var

# 假设这些键是通过安全的方式定义的，例如从配置文件或环境变量中读取。
REQUEST_ID_KEY = "X-Request-ID"
FC_REQUEST_ID_KEY = "X-Fc-Request-ID"
TRACE_ID_KEY = "traceId"


def add_middleware(app: FastAPI):
    @app.middleware('http')
    async def add_process_time_header(request: Request, call_next):
        start_time = time.perf_counter()  # 使用高精度计时器
        # 为日志添加链路ID
        # 优化请求ID的获取逻辑
        request_id = request.headers.get(REQUEST_ID_KEY,
                                         request.headers.get(FC_REQUEST_ID_KEY,
                                                             request.headers.get(TRACE_ID_KEY,
                                                                                 str(uuid.uuid4()))))
        # 确保request_id是安全的字符串
        if not isinstance(request_id, str) or not all(c.isalnum() or c in "-_" for c in request_id):
            request_id = str(uuid.uuid4())
        request_id_var.set(request_id)

        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        request_time_it_var.set(f'{process_time: .3f}')
        response.headers[f'{REQUEST_ID_KEY}_p'] = request_id
        response.headers[REQUEST_ID_KEY] = request_id
        return response

    app.add_middleware(GZipMiddleware, minimum_size=1000)
