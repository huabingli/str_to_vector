# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model_factory
   Description :
   Author :       lihb
   date：          2025/2/5
-------------------------------------------------
   Change Activity:
                   2025/2/5:
-------------------------------------------------
"""
import asyncio
import io
import threading
from abc import ABC, abstractmethod
from enum import Enum

import httpx
import torch
from PIL import Image
from httpx import AsyncClient
from loguru import logger
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor
from transformers.modeling_utils import SpecificPreTrainedModelType

from core.config import settings
from core.exceptions import AiChatException
from utils.timer import AsyncTimer


class ImageVectorizer(ABC):
    model = None
    processor = None
    device = None
    lock = threading.Lock()  # 添加锁

    @classmethod
    def get_device(cls) -> str:
        if cls.device is None:
            cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # cls.device = 'cpu'
            logger.info(f'AutoModelForZeroShotImageClassification 模型使用: {cls.device}')
        return cls.device

    @classmethod
    def initialize_model(cls):
        """
        初始化模型和处理器。
        """
        cls.get_device()
        cls.get_model()
        cls.get_processor()

    @classmethod
    @abstractmethod
    def get_model_path(cls) -> str:
        pass

    @classmethod
    def get_model(cls) -> SpecificPreTrainedModelType:
        if cls.model is None:  # 外部检查，减少锁竞争
            with cls.lock:
                if cls.model is None:  # 内部检查，防止多个线程竞争
                    model_path = cls.get_model_path()
                    cls.model = AutoModelForZeroShotImageClassification.from_pretrained(model_path,
                                                                                        torch_dtype=torch.float32)
                    cls.model.to(cls.get_device())
        return cls.model

    @classmethod
    @abstractmethod
    def get_processor_path(cls) -> str:
        pass

    @classmethod
    def get_processor(cls):
        if cls.processor is None:
            with cls.lock:
                if cls.processor is None:  # 再次检查，防止多个线程竞争
                    processor_path = cls.get_processor_path()
                    cls.processor = AutoProcessor.from_pretrained(processor_path, device=cls.get_device())
        return cls.processor

    @staticmethod
    async def fetch_image(session: AsyncClient, url):
        try:
            if url.startswith(("http://", "https://")):
                async with session.stream('GET', url) as response:
                    data = await response.aread()
                    return await asyncio.to_thread(Image.open, io.BytesIO(data))
            else:
                return await asyncio.to_thread(Image.open, url)
        except Exception as e:
            logger.error(f'图片加载失败: {url} - {e}')
            return None  # 失败的图片填充 None

    @staticmethod
    @AsyncTimer("批量图片加载")
    async def pull_images_batch(image_urls: list[str]) -> list[Image.Image]:
        """使用 `aiohttp` 进行异步批量下载并加载图片"""

        async with httpx.AsyncClient() as session:
            tasks = [ImageVectorizer.fetch_image(session, url) for url in image_urls]
            images = await asyncio.gather(*tasks)

        return images

    @classmethod
    @AsyncTimer("图片转换向量")
    async def get_image_embedding(cls, image_urls: list[str]) -> list[float]:
        """异步加载图片并获取嵌入向量"""
        images = await cls.pull_images_batch(image_urls)
        images = [img for img in images if img is not None]  # 过滤掉加载失败的图片
        if not images:
            raise AiChatException("所有图片加载失败！")
        processor = cls.get_processor()
        model = cls.get_model()

        device = cls.get_device()

        def process_images():
            with torch.no_grad():
                inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
                image_features = model.get_image_features(inputs.pixel_values)
            return image_features.tolist()

        # ✅ 让计算部分异步执行，不阻塞主线程
        vector = await asyncio.to_thread(process_images)
        if not vector or len(vector) == 0:
            raise AiChatException("图片转换向量失败，生成的特征向量为空。")
        return vector


class OpenAIClip(ImageVectorizer):
    @classmethod
    def get_model_path(cls) -> str:
        return settings.openai_clip.name_or_path

    @classmethod
    def get_processor_path(cls) -> str:
        return settings.openai_clip.name_or_path


class GoogleClip(ImageVectorizer):
    @classmethod
    def get_model_path(cls) -> str:
        return settings.google_clip.name_or_path

    @classmethod
    def get_processor_path(cls) -> str:
        return settings.google_clip.name_or_path


# Large_model_name: Literal['openai', 'google']

class LargeModelName(Enum):
    openai = OpenAIClip
    google = GoogleClip
