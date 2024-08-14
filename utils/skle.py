#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
import asyncio

from jieba import lcut
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.timer import Timer


@Timer("计算余弦相似度")
def calculate_cosine_similarity(text1, text2, subject_code: int | str = 3) -> int | float:
    """
    计算余弦相似度
    :param text1: 文章1
    :param text2: 文章2
    :param subject_code: 学科代码
    :return: 相似度
    """
    # 判断学科代码是否是英语
    if subject_code != 3 or subject_code != "3":
        vectorizer = CountVectorizer(tokenizer=lcut, token_pattern=None)
    else:
        vectorizer = CountVectorizer()

    corpus = [text1, text2]
    vectors: csr_matrix = vectorizer.fit_transform(corpus)
    # 计算余弦相似度
    similarity = cosine_similarity(vectors)
    return similarity[0][1]


async def async_cosine_similarity_process(
        text1,
        text2,
        test2_id: int = None,
        subject_code: int | str = 3
) -> (int, int | float):
    """
    使用进程池执行器,计算余弦相似度

    :param test2_id: test2 ID
    :param text1: 文章1
    :param text2: 文章2
    :param subject_code: 学科代码
    :return: 相似度
    """
    # loop = asyncio.get_event_loop()
    # return test2_id, await loop.run_in_executor(
    #         ProcessPool.get_executor(),
    #         calculate_cosine_similarity,
    #         text1, text2, subject_code
    # )
    return test2_id, await asyncio.to_thread(calculate_cosine_similarity, text1, text2, subject_code)
