#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @文件       : m3e.py
# @时间       : 2023/12/18 17:18
# @作者       : 35840
# @说明       :
import asyncio
import re
import time
from functools import lru_cache

import numpy as np
import torch
from anyio import to_thread
from loguru import logger
from sentence_transformers import SentenceTransformer

from core.config import settings
from models.acquisition_vector import AcquisitionVector2, AcquisitionVectorOutBatch
from utils.timer import AsyncTimer

re_tag = re.compile(r'<.+?>')
re_space = re.compile(r'\s+')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 使用字典处理HTML转义字符
html_escapes = {
    '&nbsp;': ' ',
    '&ensp;': ' ',
    '&emsp;': ' ',
    # '\t': ' ',
}


def escape_chars(s):
    if not s:
        return ""
    # 使用字典替换HTML转义字符
    for escape, replacement in html_escapes.items():
        s = s.replace(escape, replacement)

    s = re_tag.sub('', s)
    s = re_space.sub(' ', s)
    s = s.replace(',', '，')
    return s.strip()


# async def escape_chars(s):
#     return await to_thread.run_sync(_escape_chars, s)


class GetM3eModel:
    model: SentenceTransformer = None
    device: str = None

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls.model is None:
            cls.model = SentenceTransformer(
                    settings.m3e.name_or_path,
                    device=cls.get_device(),
                    model_kwargs={'torch_dtype': torch.float32}
            )
            logger.info(f"Model loaded from {cls.model}")
        return cls.model

    @classmethod
    def get_device(cls) -> str:
        if cls.device is None:
            cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f'SentenceTransformer 模型使用: {cls.device}')
        return cls.device

    @classmethod
    def start_model(cls):
        cls.get_model()


def np_float_to_str_to_float(s: np.float32) -> str:
    return str(s)


@lru_cache(maxsize=200)
def model_encode(article):
    """
        使用M3e模型对文章进行编码。

        参数:
            article (str): 需要被编码的文章。

        返回:
            np.float32: 文章的编码表示。

        异常:
            ValueError: 如果文章为空或格式不正确。
            Exception: 如果模型编码过程中发生错误。
    """
    article = escape_chars(article)
    start_time = time.time()
    # 获取模型实例并进行编码
    data: np.ndarray = GetM3eModel.get_model().encode([article], device=GetM3eModel.get_device(), precision='float32')
    logger.debug(f"转换vector 耗时: {(time.time() - start_time) :.5f}s ")
    # 将编码结果转换为字符串类型，再转换为float32类型，返回第一个元素
    return data.astype(np.str_).astype(np.float32)[0]


@AsyncTimer(msg="转换vector")
async def embedding_one_article(article):
    embeddings = await to_thread.run_sync(model_encode, article)
    return embeddings


def model_encode_batch(articles_content) -> list:
    # 获取模型
    model = GetM3eModel.get_model()
    device = GetM3eModel.get_device()
    start_time = time.time()
    line_embedding = model.encode(articles_content, device=device, precision='float32')
    logger.debug(f"批量转换vector 数量: {len(articles_content)} 耗时: {(time.time() - start_time) :.5f}s")
    # 将结果转换为列表
    # line_embedding_list = [[np_float_to_str_to_float(v) for v in i] for i in line_embedding]
    return line_embedding.astype(np.str_).astype(np.float32).tolist()


async def escape_chars_to(article: AcquisitionVector2):
    """
    文章转义
    :param article: 文章
    :return:
    """
    article.article = escape_chars(article.article)


@AsyncTimer(msg="批量转换vector")
async def embedding_one_article_batch(articles: list[AcquisitionVector2]) -> list[AcquisitionVectorOutBatch]:
    """
    文章列表转vector批量

    :param articles: 传入文章列表
    :return: 返回转化后的vector列表
    """

    async with asyncio.TaskGroup() as tg:
        for article in articles:
            tg.create_task(escape_chars_to(article))
    # vector_out_batch: list[AcquisitionVectorOutBatch] = []
    # 提取文章内容进行批量转换
    line_embedding = await asyncio.to_thread(model_encode_batch, [article.article for article in articles])
    # 构建转换后的结果列表
    return [
        AcquisitionVectorOutBatch(data_id=article.data_id, vector=embedding)
        for article, embedding in zip(articles, line_embedding)
    ]
