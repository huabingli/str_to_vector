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
import asyncio

from .model_factory import LargeModelName


async def async_get_image_embedding(image_url: str, model: LargeModelName = LargeModelName.openai) -> list[
    float]:
    """
     异步获取图片的向量
    :param image_url: 图片url
    :param model: 模型名称
    :return: 向量
    """
    # match model:
    #     case "openai":
    #         GetOpenaiClipModel = OpenAIClip
    #     case "google":
    #         GetOpenaiClipModel = GoogleClip
    #     case _:
    #         GetOpenaiClipModel = OpenAIClip
    image_features = await asyncio.to_thread(model.value.get_image_embedding, image_url)
    return image_features
