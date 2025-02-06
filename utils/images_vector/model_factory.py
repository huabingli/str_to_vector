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
import threading
from abc import ABC, abstractmethod
from enum import Enum

import requests
import torch
from PIL import Image
from loguru import logger
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor, BatchEncoding
from transformers.modeling_utils import SpecificPreTrainedModelType

from core.config import settings
from core.exceptions import AiChatException
from utils.timer import Timer


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
                    cls.model = AutoModelForZeroShotImageClassification.from_pretrained(model_path)
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
    @Timer("图片加载")
    def pull_images(image_url: str):
        try:
            if image_url.startswith(("http://", "https://")):
                # raise ValueError("无效的图片 URL，必须以 http:// 或 https:// 开头。")
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                image_data = response.raw
            else:
                image_data = image_url
                # with open(image_url, 'rb', encoding="UTF-8") as f:
                #     image_data = f.read()
            return Image.open(image_data)
        except Exception as e:
            raise AiChatException(f'图片加载失败: {e}')

    @classmethod
    @Timer("图片转换向量")
    def get_image_embedding(cls, image_url: str) -> list[float]:
        image = cls.pull_images(image_url)
        processor = cls.get_processor()
        model = cls.get_model()

        device = cls.get_device()
        with torch.no_grad():
            inputs: BatchEncoding = processor(images=image, return_tensors="pt", padding=True).to(device)
            # for k, v in inputs.items():
            #     inputs[k] = v.to(device)
            # 将输入数据移动到设备
            image_features: torch.Tensor = model.get_image_features(inputs.pixel_values)

        # 转换为列表
        vector = image_features.tolist()
        if not vector or len(vector[0]) == 0:
            raise AiChatException("图片转换向量失败，生成的特征向量为空。")
        return vector[0]


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
