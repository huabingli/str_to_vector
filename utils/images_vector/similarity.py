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

from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from core.exceptions import AiChatException
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

    # 获取所有图片URL
    all_image_urls = [images.image_url] + [image.image_url for image in images.batch]

    # 并行获取所有图片的向量
    try:
        embedding_results = await async_get_image_embedding(all_image_urls, model)
    except AiChatException:
        raise
    except Exception as e:
        logger.error(f"图片嵌入计算异常: {e}")
        raise AiChatException(f"图片嵌入计算异常: {e}")

    base_vector = embedding_results[0]
    other_vectors = embedding_results[1:]

    async def compute_similarity(_, vector):
        return await asyncio.to_thread(image_cosine_similarity, base_vector, vector)

        # ✅ 并发计算相似度，不阻塞

    similarity_tasks = [compute_similarity(image, vector) for image, vector in zip(images.batch, other_vectors)]
    similarity_values = await asyncio.gather(*similarity_tasks)

    # 组织结果
    # 组织结果
    similarity_list = [
        Similarity(
                aid=image.aid,
                image_bytes=image.image_bytes,
                image_url=image.image_url,
                similarity=similarity
        )
        for image, similarity in zip(images.batch, similarity_values)
    ]

    return ImageSimilarityOutBatch(similarity=similarity_list)
