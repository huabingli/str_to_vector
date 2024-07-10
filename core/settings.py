#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @文件       :settings.py
# @时间       :2023/9/21 上午11:04
# @作者       :lihb
# @说明       :
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class M3e(BaseModel):
    name_or_path: Optional[str] = Field(
            r'C:\Users\35840\PycharmProjects\moka-ai_m3e-base',
            alias='model_name_or_path',
            description="m3e模型的目录"
    )


class Base(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter='__', env_file=None)

    base_dir: Path = Path(__file__).resolve().parent.parent
    log_level: str = 'INFO'
    log_colorize: bool = Field(True, description='是否打印彩色日志')

    @property
    def base_dir_str(self) -> str:
        return str(self.base_dir)

    def update_data(self, data: dict):
        try:
            updated_model = self.model_validate(data)
        except ValidationError as e:
            raise ValueError(f"Invalid data: {e.errors()}")

        for field_name, field_value in updated_model:
            if hasattr(self, field_name):
                setattr(self, field_name, field_value)


class Settings(Base):
    m3e: M3e = M3e()
