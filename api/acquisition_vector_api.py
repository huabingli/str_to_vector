#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @文件       : acquisition_vector_api.py
# @时间       : 2023/12/25 9:10
# @作者       : lihb
# @说明       :

from fastapi import APIRouter

from models.acquisition_vector import (AcquisitionVector, AcquisitionVector2, AcquisitionVectorOut,
                                       AcquisitionVectorOutBatch)
from schemas.base import R
from utils.m3e import embedding_one_article, embedding_one_article_batch

router = APIRouter(prefix='/acquisition_vector', tags=['向量转换'])


@router.post(
        '/',
        summary='输入字符获取到向量',
        response_model=R[AcquisitionVectorOut],
        response_model_exclude_none=True
)
async def get_acquisition_vector(article: AcquisitionVector):
    # vector = await asyncio.to_thread(embedding_one_article_no_cache, article.article)
    vector = await embedding_one_article(article.article)
    return R.success(data=AcquisitionVectorOut(vector=vector, article=article.article))


@router.post(
        '/batch/',
        summary='批量输入字符获取到向量',
        response_model=R[list[AcquisitionVectorOutBatch]],
        response_model_exclude={'article'},
        response_model_exclude_none=True,
)
async def get_acquisition_vector_batch(articles: list[AcquisitionVector2]):
    # vectors = await aseyncio.to_thread(embedding_one_article_no_cache, articles)
    vectors = await embedding_one_article_batch(articles)
    return R.success(data=vectors)
