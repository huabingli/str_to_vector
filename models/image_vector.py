from typing import Union

from pydantic import BaseModel, Field


class ImageVector(BaseModel):
    image_url: Union[str] = Field(..., description="图片url")


class ImageSimilarityBatch(ImageVector):
    batch: list[ImageVector] = Field(..., description="批量图片url")


class ImageVectorOut(ImageVector):
    vector: list[float] = Field(..., description="向量")


class Similarity(ImageVector):
    similarity: float = Field(..., description="相似度")


class ImageSimilarityOutBatch(BaseModel):
    similarity: list[Similarity] = Field([], description="相似度 {url: 相似度}")
