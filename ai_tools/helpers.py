"""
配置加载辅助函数
"""

from typing import Any


def _as_bool(value: Any, default: bool) -> bool:
    """将值转换为布尔值"""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: Any, default: float) -> float:
    """将值转换为浮点数"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
