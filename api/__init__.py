#!/usr/bin/env python3 
# -*- coding: utf-8 -*- 
"""

                     _ooOoo_
                    o8888888o
                    88" . "88
                    (| -_- |)
                     O\ = /O
                 ____/`---'\____
               .   ' \\| |// `.
                / \\||| : |||// \
              / _||||| -:- |||||- \
                | | \\\ - /// | |
              | \_| ''\---/'' | |
               \ .-\__ `-` ___/-. /
            ___`. .' /--.--\ `. . __
         ."" '< `.___\_<|>_/___.' >'"".
        | | : `- \`.;`\ _ /`;.`/ - ` : | |
          \ \ `-. \_ __\ /__ _/ .-` / /
  ======`-.____`-.___\_____/___.-`____.-'======
                     `=---='
 
  .............................................
           佛祖保佑             永无BUG

@Time    : 2024/7/10 下午1:06
@Author : 李滑冰 
@Site : www.ywcsb.vip 
@File : __init__.py.py 
@Software: PyCharm
"""
__all__ = ['router']

from fastapi import APIRouter

from api import acquisition_vector_api

router = APIRouter(prefix='/api/v1')
router.include_router(router=acquisition_vector_api.router)
