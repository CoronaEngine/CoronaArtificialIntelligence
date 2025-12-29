"""
工作流执行器

提供工作流选择、加载和执行的统一入口。
从请求中识别 function_id 并路由到对应工作流。

注意：本模块为兼容层，实际实现已移至 workflow.adapter 和 workflow.executor。
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from service.common import build_error_response
from workflow.adapter import (
    extract_function_id,
    parse_request,
    format_response,
)
from workflow.registry import get_workflow_registry
from service.context import (
    set_current_session,
    reset_current_session,
)

logger = logging.getLogger(__name__)

# 重新导出 extract_function_id，保持向后兼容
extract_function_id = extract_function_id


def execute_workflow(
    request_data: Dict[str, Any],
    *,
    interface_type: str = "image",
) -> str:
    """执行工作流

    根据请求数据中的 function_id 获取并执行对应的工作流。
    所有参数（prompt、images、bounding_box 等）均从 request_data 中解析。

    参数:
        request_data: 原始请求数据
        interface_type: 接口类型（用于构建响应）

    返回:
        JSON 格式的响应字符串
    """
    session_id = request_data.get("session_id") or "default"
    metadata = request_data.get("metadata", {})

    try:
        # 使用 adapter 解析请求创建初始状态
        initial_state = parse_request(request_data)
        function_id = initial_state["function_id"]

        logger.info(f"使用工作流模式处理请求, function_id={function_id}")

        # 获取工作流注册表
        registry = get_workflow_registry()

        # 确保工作流已发现
        if not registry.has(function_id):
            registry.discover()

        # 获取工作流
        workflow = registry.get(function_id)
        if workflow is None:
            raise ValueError(f"未找到 function_id={function_id} 对应的工作流")

        logger.debug(f"工作流初始状态: {initial_state}")

        # 设置会话上下文
        session_id = initial_state.get("session_id", session_id)
        token = set_current_session(session_id)

        try:
            final_state = workflow.invoke(initial_state)
        finally:
            reset_current_session(token)

        logger.debug(f"工作流最终状态: {final_state}")

        # 使用 adapter 格式化响应
        return format_response(final_state, interface_type=interface_type)

    except Exception as exc:
        logger.error(f"工作流执行异常: {exc}")
        return build_error_response(
            interface_type=interface_type,
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )


__all__ = [
    "extract_function_id",
    "execute_workflow",
]
