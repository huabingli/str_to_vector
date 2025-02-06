# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     similarity
   Description :
   Author :       lihb
   date：          2025/2/5
-------------------------------------------------
   Change Activity:
                   2025/2/5:
-------------------------------------------------
"""
import asyncio
from typing import cast

from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from models.image_vector import ImageSimilarityBatch, ImageSimilarityOutBatch, Similarity
from utils.timer import AsyncTimer
from .embedding import async_get_image_embedding
from .model_factory import LargeModelName


def image_cosine_similarity(vector: list[float], vector2: list[float]):
    """ 计算两张图片的余弦相似度

    :param vector: 第一个图片的向量
    :param vector2: 第二个图片的向量
    :return: 图片余弦相似度，范围在 [0, 1]
    """
    similarity = cosine_similarity([vector, vector2])
    similarity_score = similarity[0][1]

    logger.info(f"图片相似度: {similarity_score:.4f}")
    return similarity_score


@AsyncTimer("图片相似度计算")
async def async_image_calculate_cosine_similarity(
        images: ImageSimilarityBatch,
        model: LargeModelName = LargeModelName.openai
) -> ImageSimilarityOutBatch:
    """ 异步计算两张图片的余弦相似度。

    :param images: 图片url
    :param model: 模型名称
    :return: 相似度结果
    """
    batch: dict[str, dict[str, Similarity | asyncio.Task | list[float]]] = {}

    # 获取所有图片的嵌入向量任务
    async with asyncio.TaskGroup() as tg:
        base_vector_task = tg.create_task(async_get_image_embedding(images.image_url, model))
        for image in images.batch:
            batch[image.aid] = {
                'task': tg.create_task(async_get_image_embedding(image.image_url, model)),
                'similarity': image
            }

    # 获取嵌入向量的结果
    base_vector = await base_vector_task
    embedding_results = {aid: await task['task'] for aid, task in batch.items()}

    # 计算相似度
    similarity_results = {}
    async with asyncio.TaskGroup() as tg:
        for aid, vector in embedding_results.items():
            logger.info(f"计算图片相似度任务: {aid}")
            similarity_results[aid] = tg.create_task(
                    asyncio.to_thread(image_cosine_similarity, base_vector, vector)
            )
    similarity_list: list[Similarity] = []
    for aid, task in similarity_results.items():
        similarity = Similarity(
                aid=aid,
                image_bytes=batch[aid]['similarity'].image_bytes,
                image_url=batch[aid]['similarity'].image_url,
                similarity=await task
        )
        similarity_list.append(similarity)
        logger.info(f"图片相似度计算完成: {similarity.aid}")

    # 构建输出结果
    data = ImageSimilarityOutBatch(similarity=similarity_list)
    return data
