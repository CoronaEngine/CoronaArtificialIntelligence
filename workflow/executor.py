"""
工作流执行器

提供统一的工作流执行入口，负责：
1. 解析请求并创建初始 State
2. 根据 function_id 获取对应的 CompiledGraph
3. 执行工作流并捕获异常
4. 格式化输出为标准响应

使用方式:
    from workflow import run_workflow

    result = run_workflow(10101, request_data)
    if result is None:
        # function_id 未注册，fallback 到原有路径
        result = handle_image_generation(request_data)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .adapter import (
    parse_request,
    format_response,
)
from .registry import get_workflow_registry
from .state import WorkflowState
from ..tools.common import build_error_response
from ..tools.context import (
    set_current_session,
    reset_current_session,
)

logger = logging.getLogger(__name__)


def run_workflow(
    function_id: int,
    request_data: Any,
    *,
    interface_type: str = "image",
) -> Optional[str]:
    """执行指定的工作流

    根据 function_id 查找已注册的工作流，执行并返回结果。
    若 function_id 未注册，返回 None（调用方可 fallback 到原有路径）。

    Args:
        function_id: 功能 ID (如 10101, 10102, 10103)
        request_data: 原始请求数据
        interface_type: 接口类型（用于响应格式化）

    Returns:
        成功时返回标准三层 JSON 响应字符串，
        function_id 未注册时返回 None
    """
    registry = get_workflow_registry()

    # 检查是否已注册
    graph = registry.get(function_id)
    if graph is None:
        logger.debug(f"Workflow not registered for function_id={function_id}")
        return None

    # 解析请求
    try:
        state = parse_request(request_data)
    except Exception as e:
        logger.error(f"Failed to parse workflow request: {e}")
        return build_error_response(
            interface_type=interface_type,
            session_id=None,
            exc=e,
        )

    # 设置会话上下文
    session_id = state.get("session_id", "default")
    token = set_current_session(session_id)

    try:
        logger.info(
            f"Running workflow function_id={function_id}, session={session_id}"
        )

        # 执行工作流
        final_state: WorkflowState = graph.invoke(state)

        # 格式化输出
        return format_response(final_state, interface_type=interface_type)

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return build_error_response(
            interface_type=interface_type,
            session_id=session_id,
            exc=e,
            metadata=state.get("metadata", {}),
        )
    finally:
        reset_current_session(token)


def run_workflow_from_request(
    request_data: Any,
    *,
    interface_type: str = "image",
) -> Optional[str]:
    """从请求中提取 function_id 并执行工作流

    便捷方法，自动从 request_data 中解析 function_id。

    Args:
        request_data: 原始请求数据
        interface_type: 接口类型

    Returns:
        成功时返回响应 JSON，未找到 function_id 或未注册时返回 None
    """
    from tools.common import (
        ensure_dict,
        extract_parameter,
    )

    data = ensure_dict(request_data)
    function_id = extract_parameter(data, "function_id")

    if function_id is None:
        logger.debug("No function_id found in request")
        return None

    # 转换为 int
    if isinstance(function_id, str):
        try:
            function_id = int(function_id)
        except ValueError:
            logger.warning(f"Invalid function_id format: {function_id}")
            return None

    return run_workflow(function_id, request_data, interface_type=interface_type)


__all__ = ["run_workflow", "run_workflow_from_request"]
