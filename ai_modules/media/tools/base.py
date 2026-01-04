"""
媒体服务基础模块

提供媒体生成服务的通用抽象和工具函数。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ai_config.ai_config import get_ai_config, AIConfig
from ai_tools.common import (
    ensure_dict,
    build_error_response,
    build_success_response,
    parse_tool_response,
)
from ai_tools.concurrency import session_concurrency
from ai_tools.workflow_executor import (
    extract_function_id,
    execute_workflow,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 通用工具函数
# ============================================================================


def extract_prompt_from_llm_content(data: Dict[str, Any]) -> str:
    """从 llm_content 中提取 prompt 文本。

    Args:
        data: 请求数据字典

    Returns:
        提取的 prompt 字符串
    """
    llm_content = data.get("llm_content")
    if not isinstance(llm_content, list) or not llm_content:
        return ""
    first = llm_content[0]
    parts = first.get("part", [])
    prompt = "".join(
        p.get("content_text", "") for p in parts if p.get("content_type") == "text"
    ).strip()
    return prompt


def extract_images_from_request(request_data: Dict[str, Any]) -> List[str]:
    """从 llm_content 中提取图片 URL 列表。

    规则：
    1. 遍历 llm_content[0]["part"] 中的所有 image/detection 类型 part。
    2. 按顺序收集所有图片 URL。

    Args:
        request_data: 请求数据字典

    Returns:
        图片 URL 列表
    """
    llm_content = request_data.get("llm_content", [])
    if not isinstance(llm_content, list) or not llm_content:
        return []

    parts = llm_content[0].get("part", [])
    image_urls: List[str] = []

    for part in parts:
        content_type = part.get("content_type")
        if content_type not in ("image", "detection"):
            continue
        url = part.get("content_url")
        if url:
            image_urls.append(url)

    return image_urls


# ============================================================================
# 媒体服务基类
# ============================================================================


class MediaServiceBase(ABC):
    """
    媒体生成服务基类

    提供统一的请求处理流程：
    1. 解析请求数据
    2. 检查是否使用工作流
    3. 并发控制
    4. 调用具体实现
    5. 构建响应

    子类需要实现：
    - interface_type: 接口类型（image/video/audio等）
    - load_tools(): 加载对应的工具
    - execute_tool(): 执行工具调用
    - clean_parts(): 清洗响应 parts（可选覆盖）
    """

    @property
    @abstractmethod
    def interface_type(self) -> str:
        """接口类型标识"""
        pass

    @abstractmethod
    def load_tools(self, cfg: AIConfig) -> List[Any]:
        """加载服务工具

        Args:
            cfg: AI 配置

        Returns:
            工具列表
        """
        pass

    @abstractmethod
    def execute_tool(
        self,
        tool: Any,
        request_data: Dict[str, Any],
        session_id: str,
        cfg: AIConfig,
    ) -> str:
        """执行工具调用

        Args:
            tool: 工具实例
            request_data: 请求数据
            session_id: 会话ID
            cfg: AI 配置

        Returns:
            工具返回的 JSON 字符串
        """
        pass

    def clean_parts(self, parts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清洗响应 parts

        默认实现：保留 content_type, content_url, content_text, url_expire_time

        Args:
            parts: 原始 parts 列表

        Returns:
            清洗后的 parts 列表
        """
        cleaned_parts = []
        for part in parts:
            cleaned_part = {
                "content_type": part.get("content_type"),
                "content_url": part.get("content_url"),
                "content_text": part.get("content_text", ""),
            }
            # 保留 url_expire_time 字段
            if "url_expire_time" in part:
                cleaned_part["url_expire_time"] = part["url_expire_time"]
            # 保留 parameter（子类可覆盖实现更严格的过滤）
            if "parameter" in part:
                cleaned_part["parameter"] = part["parameter"]

            # 移除 None 值字段
            cleaned_part = {k: v for k, v in cleaned_part.items() if v is not None}
            # 当 content_text 为空字符串时移除
            if cleaned_part.get("content_text") == "":
                cleaned_part.pop("content_text", None)
            cleaned_parts.append(cleaned_part)
        return cleaned_parts

    def handle(self, payload: Any) -> str:
        """统一处理入口

        路由逻辑：
        - 如果请求包含 function_id → 调用对应的工作流
        - 如果没有 function_id → 使用直接工具调用模式

        Args:
            payload: 请求数据

        Returns:
            JSON 格式的响应字符串
        """
        request_data: Dict[str, Any] = ensure_dict(payload)
        session_id = request_data.get("session_id") or "default"
        metadata = request_data.get("metadata", {})
        cfg = get_ai_config()

        # 提取 function_id，决定是否使用工作流
        function_id = extract_function_id(request_data)

        # 使用统一的并发控制
        with session_concurrency(session_id, cfg) as acquired:
            if not acquired:
                return build_error_response(
                    interface_type=self.interface_type,
                    session_id=session_id,
                    metadata=metadata,
                    exc=RuntimeError("并发繁忙，请稍后重试"),
                )

            # 根据是否有 function_id 选择处理路径
            if function_id is not None:
                return execute_workflow(
                    request_data=request_data,
                    interface_type=self.interface_type,
                )
            else:
                return self._handle_inner(request_data, session_id, metadata, cfg)

    def _handle_inner(
        self,
        request_data: Dict[str, Any],
        session_id: str,
        metadata: Dict[str, Any],
        cfg: AIConfig,
    ) -> str:
        """内部处理实现（在并发控制内执行）

        Args:
            request_data: 请求数据
            session_id: 会话ID
            metadata: 元数据
            cfg: AI 配置

        Returns:
            JSON 格式的响应字符串
        """
        try:
            logger.debug(f"收到 {self.interface_type} 请求: {request_data}")

            # 加载工具
            tools = self.load_tools(cfg)
            if not tools:
                raise RuntimeError(f"{self.interface_type} 功能未启用或配置不完整")

            tool = tools[0]

            # 执行工具调用
            from ai_tools.common import session_context

            with session_context(session_id) as sid:
                result_json = self.execute_tool(tool, request_data, sid, cfg)
                session_id = sid

            logger.debug(f"{self.interface_type} 工具返回: {result_json}")

            # 解析工具返回
            tool_envelope = parse_tool_response(result_json)

            # 检查错误
            if tool_envelope.get("error_code", 0) != 0:
                error_msg = tool_envelope.get("status_info", "未知错误")
                raise RuntimeError(f"{self.interface_type} 生成失败: {error_msg}")

            # 提取 llm_content
            llm_content = tool_envelope.get("llm_content", [])
            if not llm_content:
                raise RuntimeError(f"{self.interface_type} 未返回有效内容")

            # 提取并清洗 parts
            original_parts = llm_content[0].get("part", [])
            cleaned_parts = self.clean_parts(original_parts)

            if not cleaned_parts:
                raise RuntimeError(f"{self.interface_type} 未返回有效的内容部分")

            # 解析 parts 中的 fileid:// URL
            from ai_tools.response_adapter import (
                resolve_parts,
            )

            try:
                cleaned_parts = resolve_parts(cleaned_parts, timeout=150.0)
            except Exception as e:
                logger.error(f"解析 parts 中的 file_id 失败: {e}")
                raise RuntimeError(f"媒体资源解析失败: {e}") from e

            return build_success_response(
                interface_type=self.interface_type,
                session_id=session_id,
                metadata=metadata,
                parts=cleaned_parts,
            )
        except Exception as exc:
            logger.error(f"{self.interface_type} 异常: {exc}")
            return build_error_response(
                interface_type=self.interface_type,
                session_id=session_id,
                metadata=metadata,
                exc=exc,
            )


__all__ = [
    "MediaServiceBase",
    "extract_prompt_from_llm_content",
    "extract_images_from_request",
]
