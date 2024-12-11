import asyncio

import requests
import torch
from PIL import Image
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor, BatchEncoding
from transformers.modeling_utils import SpecificPreTrainedModelType

from core.config import settings
from core.exceptions import AiChatException
from models.image_vector import ImageSimilarityBatch, ImageSimilarityOutBatch, Similarity
from utils.timer import AsyncTimer, Timer


# https://huggingface.co/openai/clip-vit-base-patch32?library=transformers
# https://huggingface.co/openai/clip-vit-large-patch14?library=transformers

class GetOpenaiClipModel:
    model = None
    processor = None
    device = None

    @classmethod
    def get_device(cls) -> str:
        if cls.device is None:
            cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    def get_model(cls) -> SpecificPreTrainedModelType:
        if cls.model is None:
            cls.model = AutoModelForZeroShotImageClassification.from_pretrained(settings.openai_clip.name_or_path)
            cls.model.to(cls.get_device())
        return cls.model

    @classmethod
    def get_processor(cls):
        if cls.processor is None:
            cls.processor = AutoProcessor.from_pretrained(settings.openai_clip.name_or_path, device=cls.get_device())
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
            inputs: BatchEncoding = processor(images=image, return_tensors="pt", padding=True)
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            # 将输入数据移动到设备
            image_features: torch.Tensor = model.get_image_features(inputs.pixel_values)

        # 转换为列表
        vector = image_features.tolist()
        if not vector or len(vector[0]) == 0:
            raise AiChatException("图片转换向量失败，生成的特征向量为空。")
        return vector[0]


async def async_get_image_embedding(image_url: str) -> list[float]:
    image_features = await asyncio.to_thread(GetOpenaiClipModel.get_image_embedding, image_url)
    return image_features


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
async def async_image_calculate_cosine_similarity(images: ImageSimilarityBatch) -> ImageSimilarityOutBatch:
    """ 异步计算两张图片的余弦相似度。

    :param images: 图片url
    :return: 相似度结果
    """
    batch: dict[str, dict[str, Similarity | asyncio.Task | list[float]]] = {}

    # 获取所有图片的嵌入向量任务
    async with asyncio.TaskGroup() as tg:
        base_vector_task = tg.create_task(async_get_image_embedding(images.image_url))
        for image in images.batch:
            batch[image.aid] = {
                'task': tg.create_task(async_get_image_embedding(image.image_url)),
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
    # return await asyncio.to_thread(image_calculate_cosine_similarity, image_url, image_url2)


async def main():
    image_features = await async_get_image_embedding(
            'https://zw-cmdb.oss-cn-beijing.aliyuncs.com/test/ocr/Snipaste_2024-11-27_15-40-34.png')
    print(image_features)


if __name__ == '__main__':
    asyncio.run(main())
