# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     m3e_test
   Description :
   Author :       lihb
   date：          2025/4/25
-------------------------------------------------
   Change Activity:
                   2025/4/25:
-------------------------------------------------
"""
from pathlib import Path

import numpy as np
from elasticsearch import Elasticsearch
from sentence_transformers import quantize_embeddings

from utils.m3e import GetM3eModel


def fetch_all_vectors_merged() -> np.ndarray:
    # 判断是否有缓存文件如果有直接加载返回，如果没有则获取
    if Path("all_vectors.npy").exists():
        return np.load("all_vectors.npy")

    es = Elasticsearch(
            hosts=['http://es-cn-27a3ni2kc000bbdj1.public.elasticsearch.aliyuncs.com:9200'],
            basic_auth=('stw', 'ccglE9y4=LVwNeSBLxbJ')
    )

    all_vectors = []

    for subject in [1, 2, 3, 5, 56, 57, 6, 7, 8, 9, 4]:
        for grade in [10, 11, 12, 13, 7, 8, 9]:
            query = {
                "_source": ["questionVector"],
                "size": 500,
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"subjectCode": subject}},
                            {"term": {"gradeCode": grade}}
                        ]
                    }
                }
            }
            res = es.search(index="stw_question", body=query)
            vectors = [
                hit["_source"]["questionVector"]
                for hit in res["hits"]["hits"]
                if "questionVector" in hit["_source"]
            ]
            if vectors:
                all_vectors.extend(vectors)
                print(f"{subject}_{grade}: Fetched {len(vectors)} vectors")
        # 合并为一个大的 np.ndarray
    final_array = np.array(all_vectors, dtype=np.float32)
    np.save("all_vectors.npy", final_array)
    print(f"Total vectors merged: {final_array.shape}")
    return final_array


if __name__ == '__main__':
    a = fetch_all_vectors_merged()
    float_emb = GetM3eModel.get_model().encode(['你好'], convert_to_numpy=True, precision='float32')
    b = quantize_embeddings(embeddings=float_emb, precision='int8', calibration_embeddings=a)
    print(b)
