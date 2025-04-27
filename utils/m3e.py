#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @文件       : m3e.py
# @时间       : 2023/12/18 17:18
# @作者       : 35840
# @说明       :
import asyncio
import re
import threading
import time
from typing import Literal

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer, quantize_embeddings

from core.config import settings
from models.acquisition_vector import AcquisitionVector, AcquisitionVector2, AcquisitionVectorOutBatch
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
    lock = threading.Lock()  # 添加锁
    vectors: np.ndarray = None  # 新增变量来缓存加载的向量数据

    @classmethod
    def get_model(cls) -> SentenceTransformer:
        if cls.model is None:
            with cls.lock:
                if cls.model is None:
                    cls.model = SentenceTransformer(
                            settings.m3e.name_or_path,
                            device=cls.get_device(),
                            model_kwargs={'torch_dtype': torch.float32}
                    )
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

    @classmethod
    def load_vectors(cls):
        """加载缓存的向量数据"""
        vectors_file = settings.base_dir.joinpath('utils', 'all_vectors.npy')
        if vectors_file.exists():
            cls.vectors = np.load(vectors_file)
            logger.info(f"加载缓存的向量数据: {cls.vectors.shape}")
        else:
            logger.warning(f"没有找到缓存的向量数据文件: {vectors_file}")

    @classmethod
    def get_vectors(cls) -> np.ndarray:
        """返回加载的向量数据"""
        if cls.vectors is None:
            cls.load_vectors()
        return cls.vectors


def np_float_to_str_to_float(s: np.float32) -> str:
    return str(s)


def model_encode(
        article: list[str] | str,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = 'float32'
) -> list:
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
    # 判断是否是list类型
    if isinstance(article, str):
        article = [article]

    # Escape处理
    article = [escape_chars(a) for a in article]

    model = GetM3eModel.get_model()
    device = GetM3eModel.get_device()

    start_time = time.time()
    # int8需要float32先编码，再量化
    encode_precision: Literal[
        "float32", "int8", "uint8", "binary", "ubinary"] = 'float32' if precision == 'int8' else precision

    # 获取模型实例并进行编码
    embeddings: np.ndarray = model.encode(article, device=device, precision=encode_precision)

    logger.debug(f"转换vector 耗时: {(time.time() - start_time) :.5f}s ")
    # 将编码结果转换为字符串类型，再转换为float32类型，返回第一个元素
    if precision == 'float32':
        return embeddings.astype(np.str_).tolist()
    elif precision == 'int8':
        return quantize_embeddings(
                embeddings=embeddings,
                precision='int8',
                calibration_embeddings=GetM3eModel.get_vectors()
        ).tolist()
    return embeddings.tolist()


@AsyncTimer(msg="转换vector")
async def embedding_one_article(article: AcquisitionVector):
    embeddings = await asyncio.to_thread(model_encode, article.article, article.vector_precision)
    return embeddings[0]


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

    article_list: list[str] = [article.article for article in articles]

    # 提取文章内容进行批量转换
    line_embedding = await asyncio.to_thread(
            model_encode,
            article_list,
            articles[0].vector_precision
    )
    # 构建转换后的结果列表
    return [
        AcquisitionVectorOutBatch(data_id=article.data_id, vector=embedding)
        for article, embedding in zip(articles, line_embedding)
    ]


def convert_to_ndarray(matrix: list[list[float]]) -> np.ndarray:
    """
    将向量矩阵转换成np.ndarray

    参数:
    matrix (list): 向量矩阵，每个元素是一个向量，向量由多个数值组成

    返回:
    ndarray: 转换后的np.ndarray向量矩阵
    """
    return np.array(matrix, dtype=np.float32)


if __name__ == '__main__':
    GetM3eModel.start_model()
    print(model_encode(
            ["nihao"],
            'float32'))
