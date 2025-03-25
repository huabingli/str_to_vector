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
import time
from abc import ABC, abstractmethod
from enum import Enum

import httpx
import torch
from PIL import Image
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
    _httpx_client = None  # 类级别共享客户端
    _httpx_client_lock = asyncio.Lock()

    @classmethod
    def get_device(cls) -> str:
        if cls.device is None:
            cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # cls.device = 'cpu'
            logger.info(f'AutoModelForZeroShotImageClassification 模型使用: {cls.device}')
        return cls.device

    @classmethod
    async def get_httpx_client(cls):
        """获取全局复用客户端"""
        if cls._httpx_client is None:
            async with cls._httpx_client_lock:
                if cls._httpx_client is None:  # 双重检查锁定
                    cls._httpx_client = httpx.AsyncClient(
                            timeout=httpx.Timeout(5.0, connect=2.0),
                            limits=httpx.Limits(
                                    max_keepalive_connections=50,
                                    max_connections=100
                            ),
                    )
        return cls._httpx_client

    @classmethod
    async def close_httpx(cls):
        """显式关闭客户端"""
        if cls._httpx_client:
            await cls._httpx_client.aclose()
            cls._httpx_client = None

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
                    cls.model = AutoModelForZeroShotImageClassification.from_pretrained(
                            model_path,
                            torch_dtype=torch.float32,
                    )
                    cls.model.to(cls.get_device())
                    # 仅当CUDA可用时编译模型
                    if 'cuda' in cls.get_device():
                        cls.model = torch.compile(cls.model, mode="max-autotune")
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
    async def fetch_image(url):
        try:
            if url.startswith(("http://", "https://")):
                start = time.perf_counter()
                session = await ImageVectorizer.get_httpx_client()
                response = await session.get(url, timeout=5.0)
                response.raise_for_status()
                data = await response.aread()
                logger.debug(f'图片加载成功: {url} - {time.perf_counter() - start:.3f}s')
                return await asyncio.to_thread(
                        Image.open, io.BytesIO(data)
                )
            else:
                return await asyncio.to_thread(
                        Image.open, url
                )
        except Exception as e:
            logger.error(f'图片加载失败: {url} - {e}')
            return None

    @staticmethod
    @AsyncTimer("批量图片加载")
    async def pull_images_batch(image_urls: list[str]) -> list[Image.Image]:
        """使用 `aiohttp` 进行异步批量下载并加载图片"""

        # async with httpx.AsyncClient(
        #         timeout=httpx.Timeout(5.0, connect=2.0),
        #         limits=httpx.Limits(
        #                 max_keepalive_connections=20,
        #                 max_connections=100
        #         )
        # ) as session:
        tasks = [ImageVectorizer.fetch_image(url) for url in image_urls]
        images = await asyncio.gather(*tasks)

        return images

    @staticmethod
    def process_images_with_model(images, processor, model, device):
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            features = model.get_image_features(inputs["pixel_values"])
        return features.tolist()

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

        # ✅ 让计算部分异步执行，不阻塞主线程
        vector = await asyncio.to_thread(
                cls.process_images_with_model, images, processor, model, device
        )
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
