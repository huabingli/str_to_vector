# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     embedding
   Description :
   Author :       lihb
   date：          2025/2/5
-------------------------------------------------
   Change Activity:
                   2025/2/5:
-------------------------------------------------
"""

from loguru import logger

from core.exceptions import AiChatException
from .model_factory import LargeModelName


async def async_get_image_embedding(
        image_urls: list[str],
        model: LargeModelName = LargeModelName.openai
) -> list[list[float]]:
    """
    异步获取图片的向量，支持批量处理。

    :param image_urls: 图片url列表
    :param model: 选择的模型
    :return: 每张图片对应的向量列表
    """
    if not image_urls:
        raise ValueError("图片URL列表不能为空")

    try:
        image_features = await model.value.get_image_embedding(image_urls)
        return image_features
    except Exception as e:
        logger.exception(f"获取图片嵌入向量失败: {e}")
        raise AiChatException(f"获取图片向量失败: {e}")
