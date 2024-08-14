#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
from pydantic import BaseModel, Field


class SimilarityArticles(BaseModel):
    s_id: int = Field(..., description="相似文章id", validation_alias='id')
    question_article: str = Field(..., description="相似文章", validation_alias='questionArticle')


class QuestionScoreVOsReq(BaseModel):
    article: str = Field(..., description="文章", validation_alias='article')
    similarity_articles: list[SimilarityArticles] = Field(
            ...,
            description="相似文章集合",
            validation_alias='smilarityArticles'
    )
    subject_code: int = Field(..., description="学科代码, 3是英语", validation_alias='subjectCode')
    similarity: float | int = Field(..., description="相似度", validation_alias='similarity')
