#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
import asyncio

from fastapi import APIRouter
from loguru import logger

from core.exceptions import AiChatException
from models.similarity_calculation import QuestionScoreVOsReq
from schemas.base import R
from utils.skle import async_cosine_similarity_process

router = APIRouter(prefix='/similarity_calculation', tags=['相似题计算'])


@router.post(
        '/cosine_similarity',
        summary='余弦相似度计算',
        response_model_exclude_none=True,
        response_model=R[dict[str, float | int]]
)
async def async_cosine_similarity(question_score_vos_req: QuestionScoreVOsReq) -> R[dict[str, float | int]]:
    """
    余弦相似度计算
    :param question_score_vos_req: 文章相似度计算请求
    :return: 文章相似度计算结果
    """
    data = {}

    for task in asyncio.as_completed(
            [async_cosine_similarity_process(
                    question_score_vos_req.article,
                    question_score_vo.question_article,
                    test2_id=question_score_vo.s_id,
                    subject_code=question_score_vos_req.subject_code
            ) for question_score_vo in question_score_vos_req.similarity_articles]
    ):
        try:
            s_id, score = await task
            logger.info(f'异步任务执行成功: {s_id}, {score}, {question_score_vos_req.similarity}')
            if score > question_score_vos_req.similarity:
                data[str(s_id)] = score
        except Exception as e:
            logger.error(f'异步任务执行失败: {e}')
            raise AiChatException(f'处理查询时发生错误: {e}')
    return R.success(data=data)
