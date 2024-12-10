import random
import time
from pathlib import Path
from typing import Self

from pydantic import Base64Bytes, BaseModel, Field, model_validator, AliasChoices

tmp = Path(r'D:\PycharmProjects\str_to_vector\tmp')


class ImageVector(BaseModel):
    aid: str | int | None = Field(default_factory=lambda: int(time.time() * 1000), description="图片id",
                                  validation_alias=AliasChoices('id', 'aid'), serialization_alias='id')
    image_url: str | None = Field(None, description="图片url")
    image_bytes: Base64Bytes | None = Field(None, description="图片bytes")

    @model_validator(mode='after')
    def image_bytes_or_url(self) -> Self:
        if not self.image_bytes and not self.image_url:
            raise ValueError("image_bytes or image_url is required")
        if self.image_bytes is None and self.image_url:
            return self
        # decoded_image = base64.b64decode(self.image_bytes)
        image_path = tmp.joinpath(f"{self.aid or str(random.random())}.jpg")
        with open(image_path, 'wb') as f:
            f.write(self.image_bytes)
        self.image_url = str(image_path)
        return self


class ImageSimilarityBatch(ImageVector):
    batch: list[ImageVector] = Field(..., description="批量图片url")


class ImageVectorOut(ImageVector):
    vector: list[float] = Field(..., description="向量")


class Similarity(ImageVector):
    similarity: float = Field(..., description="相似度")


class ImageSimilarityOutBatch(BaseModel):
    similarity: list[Similarity] = Field([], description="相似度 {url: 相似度}")

    def sort_by_similarity(self, reverse: bool = True):
        self.similarity.sort(key=lambda x: x.similarity, reverse=reverse)
