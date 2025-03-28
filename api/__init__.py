#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

__all__ = ['router']

from fastapi import APIRouter

from api import acquisition_vector_api, image_vector_api, similarity_calculation_api

router = APIRouter(prefix='/api/v1')
router.include_router(router=acquisition_vector_api.router)
router.include_router(router=similarity_calculation_api.router)

router.include_router(router=image_vector_api.router)

@router.get('/heart/check', tags=['heart'], summary='探测接口')
async def get_heart_check() -> dict:
    return {'status': 'ok'}