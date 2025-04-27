# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test_cos_sim
   Description :
   Author :       lihb
   date：          2025/4/25
-------------------------------------------------
   Change Activity:
                   2025/4/25:
-------------------------------------------------
"""
from datetime import datetime
from typing import Iterator, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
from loguru import logger
from pydantic import BaseModel, Field
from sentence_transformers import quantize_embeddings

from utils.m3e import GetM3eModel, convert_to_ndarray

BATCH_SIZE = 1000

es = Elasticsearch(
        hosts=['http://es-cn-27a3ni2kc000bbdj1.public.elasticsearch.aliyuncs.com:9200'],
        basic_auth=('stw', 'ccglE9y4=LVwNeSBLxbJ')
)


class QuestionDocument(BaseModel):
    content_length: int = Field(..., alias='contentLength')
    index: str = Field('stw_question_2024_07_16_v3', alias='_index')
    index_id: int = Field(..., validation_alias='id', serialization_alias='_id')
    op_type: str = Field('index', alias='_op_type')
    id: Optional[int]
    grade_code: Optional[str] = Field(None, alias="gradeCode", )
    subject_code: Optional[str] = Field(None, alias="subjectCode", )
    study_phase_code: Optional[str] = Field(None, alias="studyPhaseCode", )
    textbook_version_code: Optional[str] = Field(None, alias="textbookVersionCode", )
    ceci_code: Optional[str] = Field(None, alias="ceciCode", )
    year_code: Optional[str] = Field(None, alias="yearCode", )
    term_code: Optional[str] = Field(None, alias="termCode", )
    sync_type: Optional[int] = Field(None, alias="syncType", )
    question_type_code: Optional[str] = Field(None, alias="questionTypeCode", )
    question_category: Optional[int] = Field(None, alias="questionCategory", )
    question_article: Optional[str] = Field(None, alias="questionArticle", )
    parent_id: Optional[int | str] = Field(None, alias="parentId", )
    create_time: Optional[datetime] = Field(None, alias="createTime", )
    question_vector: list[float | int] = Field([], alias="questionVector", )


def fetch_all_documents() -> Iterator[QuestionDocument]:
    for _doc in scan(es, query={"query": {"match_all": {}}}, index='stw_question', scroll='5m', size=10000):
        yield QuestionDocument.model_validate(_doc['_source'])


def conversion(data: QuestionDocument) -> QuestionDocument:
    vectors = convert_to_ndarray([data.question_vector])
    data.question_vector = quantize_embeddings(
            embeddings=vectors,
            precision='int8',
            calibration_embeddings=GetM3eModel.get_vectors()
    ).tolist()[0]
    return data


def batch_conversion(batch: list[QuestionDocument]) -> list[QuestionDocument]:
    vectors = [doc.question_vector for doc in batch]
    ndarray = convert_to_ndarray(vectors)
    quantized_vectors = quantize_embeddings(
            embeddings=ndarray,
            precision='int8',
            calibration_embeddings=GetM3eModel.get_vectors()
    ).tolist()
    for doc, vec in zip(batch, quantized_vectors):
        doc.question_vector = vec
    return batch


def bulk_write_batch(batch: list[QuestionDocument]):
    try:
        start_time = datetime.now()
        logger.debug(f"批处理开始: {len(batch)}")
        converted = batch_conversion(batch)
        actions = [doc.model_dump(by_alias=True, exclude={'question_article'}) for doc in converted]
        bulk(es, actions)
        logger.debug(f"批处理耗时: {(datetime.now() - start_time).total_seconds()}s")
    except Exception as e:
        logger.exception(e)


def bulk_write(data: QuestionDocument):
    bulk(es, [data.model_dump(by_alias=True, exclude={'question_article'})])


if __name__ == '__main__':
    pass
    # for _doc in fetch_all_documents():
    #     _data = conversion(_doc)
    #     bulk_write(_data)
    # buffer = []
    # with ThreadPoolExecutor(max_workers=100) as executor:
    #     for doc_ in fetch_all_documents():
    #         buffer.append(doc_)
    #
    #         if len(buffer) >= BATCH_SIZE:
    #             executor.submit(bulk_write_batch, buffer)
    #             buffer = []
    #
    #     if buffer:
    #         executor.submit(bulk_write_batch, buffer)
