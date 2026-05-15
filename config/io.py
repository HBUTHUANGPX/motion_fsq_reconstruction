"""YAML 配置加载工具。

职责：
    将 YAML 文件递归转换为 `MotionFSQReconstructionConfig`。
前置条件：
    YAML 顶层是 mapping。
后置条件：
    未出现的字段使用 dataclass 默认值，未知字段被忽略。
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .schema import MotionFSQReconstructionConfig

T = TypeVar("T")


def load_config(path: str | Path) -> MotionFSQReconstructionConfig:
    """加载 YAML 配置。

    前置条件：
        `path` 指向存在的 YAML 文件。
    后置条件：
        返回完整配置对象。
    """

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} 必须包含 YAML mapping。")
    return _from_dict(MotionFSQReconstructionConfig, raw)


def _from_dict(cls: type[T], values: dict[str, Any]) -> T:
    kwargs: dict[str, Any] = {}
    prototype = cls()
    for field_info in fields(cls):
        if field_info.name not in values:
            continue
        value = values[field_info.name]
        default_value = getattr(prototype, field_info.name)
        if is_dataclass(default_value) and isinstance(value, dict):
            kwargs[field_info.name] = _from_dict(type(default_value), value)
        else:
            kwargs[field_info.name] = value
    return cls(**kwargs)
