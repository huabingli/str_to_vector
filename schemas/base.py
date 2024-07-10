#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @文件       :base.py
# @时间       :2023/9/22 下午5:00
# @作者       :lihb
# @说明       :
from typing import Generic, Optional, TypeVar, Union

from pydantic import BaseModel, Field

T = TypeVar('T')


class R(BaseModel, Generic[T]):
    code: int = Field(200, description='返回状态')
    data: Optional[T] = Field(None, description='返回的数据')
    msg: str
    err: Optional[Union[str, dict, list]] = Field(None, description='错误信息。平常为空，只有报错才有')

    @classmethod
    def success(cls, msg: str = '成功', data: T = None, code: int = 200, ) -> "R":
        # 创建一个成功响应实例
        return cls(code=code, msg=msg, data=data)

    @classmethod
    def fail(cls, msg: str = '失败', data: T = None, code: int = 400, err: str | dict | list = None) -> "R":
        # 创建一个失败响应实例
        return cls(code=code, msg=msg, data=data, err=err)

    @classmethod
    def unauthorized(cls, msg: str = '认证失败', data: T = None, err: str | dict | list = None) -> "R":
        # 创建一个未授权响应实例
        return cls(code=401, msg=msg, data=data, err=err)

    @classmethod
    def empty(cls, msg: str = '查询结果为空', data: T = None, err: str | dict | list = None) -> "R":
        # 创建一个查询结果为空地响应实例
        return cls(code=404, msg=msg, data=data, err=err)

    @classmethod
    def unique(cls, msg: str = '数据已存在', data: T = None, err: str | dict | list = None) -> "R":
        # 创建一个数据已存在的响应实例
        return cls(code=404, msg=msg, data=data, err=err)

    class Config:
        none = "ignore"
        str_strip_whitespace = True
