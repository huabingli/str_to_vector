#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @文件       : acquisition_vector.py.py
# @时间       : 2023/12/25 9:22
# @作者       : lihb
# @说明       :
from typing import Union

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from core.exceptions import AiChatException


class AcquisitionVector(BaseModel):
    article: Union[str, int, float] = Field(..., description="文章")

    @field_validator("article", mode='before')
    @classmethod
    def validate_article(cls, article: Union[int, float, str]) -> str:
        if not isinstance(article, str):
            try:
                article = str(article)
                logger.debug(f"{article} 转换为str类型")
            except AttributeError:
                raise AiChatException(message=f'{article} 转换字符串出现错误')
        return article


class AcquisitionVectorOut(AcquisitionVector):
    vector: list[float] = Field(..., description="向量")


class AcquisitionVector2(AcquisitionVector):
    data_id: str | int = Field(..., description="文章ID")


class AcquisitionVectorOutBatch(BaseModel):
    vector: list[float] = Field([], description="向量")
    data_id: str | int = Field(..., description="文章ID")
