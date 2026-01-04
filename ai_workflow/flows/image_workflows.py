"""
图像工作流示例

提供一个简单的图像生成工作流示例，供参考和测试。

本模块保留一个基础示例，演示工作流的构建方式。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, TYPE_CHECKING

from langgraph.graph import StateGraph, START, END

from ai_workflow.state import WorkflowState
from ai_workflow.nodes import (
    make_tool_node,
    make_error_check_node,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


# ===========================================================================
# 工具加载（惰性）
# ===========================================================================

_cached_image_tool = None


def _get_image_tool():
    """惰性加载图像生成工具"""
    global _cached_image_tool
    if _cached_image_tool is None:
        from ai_tools.registry import get_tool_registry

        registry = get_tool_registry()
        tool = registry.get("generate_image")

        if tool is not None:
            _cached_image_tool = tool
        else:
            # 回退：直接加载
            from ai_config.ai_config import get_ai_config
            from ai_modules.image_generate.tools.image_tools import (
                load_image_tools,
            )

            config = get_ai_config()
            tools = load_image_tools(config)
            if tools:
                _cached_image_tool = tools[0]
            else:
                raise RuntimeError("Image generation tool not available")
    return _cached_image_tool


# ===========================================================================
# 示例工作流
# ===========================================================================


def _image_arg_mapper(state: WorkflowState) -> Dict[str, Any]:
    """从 State 提取图像生成工具参数"""
    return {
        "prompt": state.get("prompt", ""),
        "resolution": state.get("resolution", "1:1"),
        "image_urls": state.get("images", []) or None,
        "image_size": state.get("image_size", "2K"),
    }


def build_simple_image_workflow() -> "CompiledStateGraph":
    """构建简单图像生成工作流示例

    流程: START → generate → check → END

    这是一个最小化示例，仅包含：
    1. 调用图像生成工具
    2. 检查结果

    更复杂的业务工作流请参考 InnerAgentWorkflow/ai_workflows/
    """
    graph = StateGraph(WorkflowState)

    def generate_node(state: WorkflowState) -> Dict[str, Any]:
        """调用图像生成工具"""
        if state.get("error"):
            return {}
        try:
            tool = _get_image_tool()
            node_fn = make_tool_node(tool, _image_arg_mapper)
            return node_fn(state)
        except Exception as e:
            logger.error(f"Generate node failed: {e}")
            return {"error": str(e)}

    graph.add_node("generate", generate_node)
    graph.add_node("check", make_error_check_node())

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "check")
    graph.add_edge("check", END)

    return graph.compile()


# ===========================================================================
# 导出工作流
# ===========================================================================

# 注意：业务工作流 (10101/10102/10103) 已迁移至 InnerAgentWorkflow
# 此处仅导出示例工作流，使用 function_id 10000 表示测试/示例
WORKFLOWS: Dict[int, "CompiledStateGraph"] = {
    10000: build_simple_image_workflow(),  # 示例工作流
}

__all__ = ["WORKFLOWS", "build_simple_image_workflow"]
