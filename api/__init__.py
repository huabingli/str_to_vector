#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

__all__ = ['router']

from fastapi import APIRouter

from api import acquisition_vector_api, similarity_calculation_api

router = APIRouter(prefix='/api/v1')
router.include_router(router=acquisition_vector_api.router)
router.include_router(router=similarity_calculation_api.router)
