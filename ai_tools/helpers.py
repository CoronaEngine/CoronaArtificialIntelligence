"""
配置加载辅助函数
"""

import datetime
import logging
from typing import Any

logger = logging.getLogger(__name__)


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


def request_time_diff(payload: Any):
    # 超时处理
    # 检查请求时间是否超过240秒
    if payload.get("start_datetime"):
        now = datetime.datetime.now()
        request_time = datetime.datetime.strptime(payload.get("start_datetime"), "%Y-%m-%d %H:%M:%S")

        # 计算时间差（秒）
        time_diff = (now - request_time).total_seconds()
        # 如果时间差超过260秒，直接返回失败响应
        if time_diff > 260:
            logger.warning(
                f"请求已过期:"
                f"time_diff={time_diff:.2f}s, "
                f"request_time={request_time}, "
                f"current_time={now}"
            )

            # 返回错误响应
            raise RuntimeError(f"任务预期超时: 已超出时间{time_diff:.2f}s")
