"""
白底图检测服务

处理白底图检测请求，调用远程 API 进行检测并返回标准格式的响应
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ai_config.ai_config import get_ai_config
from ai_tools.common import (
    ensure_dict,
    build_error_response,
    build_success_response,
    parse_tool_response,
    session_context
)
from ai_tools.concurrency import session_concurrency
from ai_tools.session_tracking import init_session, update_session_state, set_session_error

# 日志配置
logger = logging.getLogger(__name__)

from ai_service.entrance import register_entrance

@register_entrance(handler_name="handle_whiteback_detection")
def handle_whiteback_detection(payload: Any) -> str:
    """
    白底图检测接口，返回标准三层结构。

    输入格式 (llm_content):
    - part[].content_type: "image" - 待检测的图片
    - part[].content_url: 图片 URL

    输出格式 (llm_content):
    - part[].content_type: "whiteback_detect"
    - part[].content_text: 检测结果描述（"True" 或 "False"）
    - part[].content_url: 原始图片 URL
    - part[].parameter.white_base_value: 白底值（0 || 1）
    """
    request_data: Dict[str, Any] = ensure_dict(payload)
    session_id = request_data.get("session_id") or "default"
    metadata = request_data.get("metadata", {})
    cfg = get_ai_config()

    # 使用统一的并发控制
    with session_concurrency(session_id, cfg) as acquired:
        if not acquired:
            return build_error_response(
                interface_type="whiteback_detection",
                session_id=session_id,
                metadata=metadata,
                exc=RuntimeError("并发繁忙，请稍后重试"),
            )
        return _handle_whiteback_detection_inner(
            request_data, session_id, metadata, cfg
        )


def _handle_whiteback_detection_inner(
    request_data: Dict[str, Any],
    session_id: str,
    metadata: Dict[str, Any],
    cfg,
) -> str:
    """白底图检测内部实现（在并发控制内执行）"""
    try:
        # 初始化会话追踪
        init_session(
            session_id=session_id,
            input_type="whiteback_detection",
            parameters=request_data,
        )
        update_session_state(session_id, "running")

        logger.debug(f"收到白底图检测请求: {request_data}")

        # 提取图片 URL 列表
        image_urls = _extract_image_urls(request_data.get("llm_content", []))
        if not image_urls:
            raise ValueError(
                "缺少待检测的图片，请在 llm_content 中提供 content_type 为 'image' 的图片 URL"
            )

        # 加载白底图检测工具
        from tools.whiteback_detect_tool import (
            load_whiteback_detect_tools,
        )

        tools = load_whiteback_detect_tools(cfg)
        if not tools:
            raise RuntimeError("白底图检测功能未启用或配置不完整")

        detect_tool = tools[0]

        # 对每张图片进行检测
        all_parts: List[Dict[str, Any]] = []

        with session_context(session_id) as sid:
            logger.debug(f"进入 session_context: {sid}")

            for image_url in image_urls:
                try:
                    logger.info(f"检测图片: {image_url}")

                    # 调用检测工具
                    result_json = detect_tool.func(image_url=image_url)
                    logger.debug(f"detect_tool 返回: {result_json}")

                    # 解析 Tool 返回的 Envelope JSON
                    tool_envelope = parse_tool_response(result_json)
                    logger.debug(f"解析 tool_envelope: {tool_envelope}")

                    # 检查错误
                    if tool_envelope.get("error_code", 0) != 0:
                        error_msg = tool_envelope.get("status_info", "未知错误")
                        logger.error(f"检测图片 {image_url} 失败: {error_msg}")

                        # 添加失败结果
                        all_parts.append(
                            {
                                "content_type": "whiteback_detect",
                                "content_text": "False",
                                "content_url": image_url,
                                "parameter": {
                                    "white_base_value": 0,
                                    "error": error_msg,
                                },
                            }
                        )
                        continue

                    # 提取 llm_content 中的 parts
                    llm_content = tool_envelope.get("llm_content", [])
                    if llm_content:
                        parts = llm_content[0].get("part", [])
                        all_parts.extend(parts)

                        # 日志记录
                        if parts:
                            part = parts[0]
                            logger.info(
                                f"检测完成: {part.get('content_text')}, "
                                f"white_base_value: {part.get('parameter', {}).get('white_base_value')}"
                            )
                    else:
                        logger.error(f"检测图片 {image_url} 未返回有效内容")
                        all_parts.append(
                            {
                                "content_type": "whiteback_detect",
                                "content_text": "False",
                                "content_url": image_url,
                                "parameter": {
                                    "white_base_value": 0,
                                    "error": "未返回有效内容",
                                },
                            }
                        )

                except Exception as e:
                    logger.exception(f"检测图片 {image_url} 异常: {e}")
                    all_parts.append(
                        {
                            "content_type": "whiteback_detect",
                            "content_text": "False",
                            "content_url": image_url,
                            "parameter": {
                                "white_base_value": 0,
                                "error": str(e),
                            },
                        }
                    )

            session_id = sid

        logger.info(f"检测完成，共检测 {len(all_parts)} 张图片")

        update_session_state(session_id, "completed")

        return build_success_response(
            interface_type="whiteback_detection",
            session_id=session_id,
            metadata=metadata,
            parts=all_parts,
        )

    except Exception as exc:
        logger.error(f"白底图检测异常: {exc}")
        set_session_error(session_id, str(exc))
        update_session_state(session_id, "failed")
        return build_error_response(
            interface_type="whiteback_detection",
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )


def _extract_image_urls(llm_content: List[Dict[str, Any]]) -> List[str]:
    """
    从 llm_content 中提取图片 URL。

    支持两种格式：
    1. 嵌套格式（优先）：llm_content[].part[].content_url
    2. 扁平格式：llm_content[].content_url

    Args:
        llm_content: llm_content 数组

    Returns:
        图片 URL 列表
    """
    image_urls = []

    for item in llm_content:
        # 尝试嵌套格式（优先）：llm_content[].part[].content_url
        if "part" in item and isinstance(item["part"], list):
            for part in item["part"]:
                content_type = part.get("content_type", "")
                content_url = part.get("content_url", "")

                if content_type == "image" and content_url:
                    image_urls.append(content_url)

        # 尝试扁平格式：llm_content[].content_url
        else:
            content_type = item.get("content_type", "")
            content_url = item.get("content_url", "")

            if content_type == "image" and content_url:
                image_urls.append(content_url)

    return image_urls
