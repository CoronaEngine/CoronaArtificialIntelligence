"""
白底图检测服务

处理白底图检测请求，调用远程 API 进行检测并返回标准格式的响应
"""

import json
import logging
from typing import Any, Dict, List, Optional

from InnerAgentWorkflow.ai_tools.whiteback_detect_tool import (
    get_whiteback_detect_tool,
)
from service.common import (
    build_success_response,
)
from service.entrance import register_entrance

# 日志配置
logger = logging.getLogger(__name__)


@register_entrance(handler_name="handle_whiteback_detection")
def handle_whiteback_detection(request_data: Dict[str, Any]) -> str:
    """
    处理白底图检测请求

    Args:
        request_data: 请求数据，格式为标准 JSON API 格式:
        {
            "session_id": str,  # 会话 ID（必需）
            "llm_content": [    # 内容数组（必需）
                {
                    "content_type": "image",  # 待检测的图片
                    "content_url": str,       # 图片 URL
                    ...
                }
            ],
            "metadata": dict    # 可选的元数据
        }

    Returns:
        JSON 字符串，格式为:
        {
            "session_id": str,
            "error_code": int,      # 0: 成功, 非0: 失败
            "status_info": str,     # 状态信息
            "llm_content": [        # 检测结果数组
                {
                    "content_type": "whiteback_detect",
                    "content_text": str,    # 检测结果描述
                    "content_url": str,     # 原始图片 URL
                    "parameter": {
                        "is_whiteback": bool,    # 是否为白底图
                        "confidence": float,     # 置信度 (0-1)
                    }
                }
            ],
            "metadata": dict
        }
    """
    session_id = request_data.get("session_id", "unknown")
    llm_content_input = request_data.get("llm_content", [])
    metadata = request_data.get("metadata", {})

    logger.info(
        f"[WhitebackDetection] 收到请求 | "
        f"session_id: {session_id} | "
        f"llm_content 长度: {len(llm_content_input)}"
    )

    try:
        # 验证输入
        if not llm_content_input:
            error_msg = "llm_content 为空，无法进行检测"
            logger.error(f"[WhitebackDetection] {error_msg} | session_id: {session_id}")
            return _build_error_response(
                session_id=session_id,
                error_code=400,
                error_msg=error_msg,
                metadata=metadata,
            )

        # 提取图片 URL
        image_urls = _extract_image_urls(llm_content_input)
        if not image_urls:
            error_msg = "llm_content 中未找到图片 URL"
            logger.error(f"[WhitebackDetection] {error_msg} | session_id: {session_id}")
            return _build_error_response(
                session_id=session_id,
                error_code=400,
                error_msg=error_msg,
                metadata=metadata,
            )

        # 获取白底图检测工具
        detect_tool = get_whiteback_detect_tool()

        # 对每张图片进行检测
        llm_content_output = []
        for image_url in image_urls:
            try:
                logger.info(
                    f"[WhitebackDetection] 检测图片 | "
                    f"session_id: {session_id} | "
                    f"image_url: {image_url}"
                )

                # 调用检测工具（返回 ToolResult）
                detection_result = detect_tool.detect(
                    image_url=image_url,
                    session_id=session_id,
                    metadata=metadata,
                )

                # 直接使用工具返回的 parts（工具层已输出最终格式）
                if detection_result.parts:
                    llm_content_output.extend(detection_result.parts)

                    # 日志记录
                    part = detection_result.parts[0]
                    logger.info(
                        f"[WhitebackDetection] 检测完成 | "
                        f"session_id: {session_id} | "
                        f"content_text: {part.get('content_text')} | "
                        f"white_base_value: {part.get('parameter', {}).get('white_base_value')}"
                    )

            except Exception as e:
                logger.exception(
                    f"[WhitebackDetection] 检测单张图片失败 | "
                    f"session_id: {session_id} | "
                    f"image_url: {image_url} | "
                    f"错误: {e}"
                )

                # 添加失败结果
                output_item = {
                    "content_type": "whiteback_detect",
                    "content_text": "False",
                    "content_url": image_url,
                    "parameter": {
                        "white_base_value": 0.0,
                        "error": str(e),
                    },
                }
                llm_content_output.append(output_item)

        logger.info(
            f"[WhitebackDetection] 请求处理完成 | "
            f"session_id: {session_id} | "
            f"检测图片数: {len(llm_content_output)}"
        )

        # 使用统一的响应构建函数（通过 llm_content 参数避免重复解析）
        return build_success_response(
            interface_type="whiteback_detection",
            session_id=session_id,
            parts=None,  # 不使用 parts
            metadata=metadata,
            llm_content=[
                {
                    "role": "assistant",
                    "interface_type": "whiteback_detection",
                    "sent_time_stamp": int(__import__("time").time()),
                    "part": llm_content_output,
                }
            ],
        )

    except Exception as e:
        logger.exception(
            f"[WhitebackDetection] 请求处理失败 | "
            f"session_id: {session_id} | "
            f"错误: {e}"
        )
        return _build_error_response(
            session_id=session_id,
            error_code=500,
            error_msg=f"服务器内部错误: {str(e)}",
            metadata=metadata,
        )


def _extract_image_urls(llm_content: List[Dict[str, Any]]) -> List[str]:
    """
    从 llm_content 中提取图片 URL

    支持两种格式：
    1. 扁平格式：llm_content[].content_url
    2. 嵌套格式：llm_content[].part[].content_url (BackGroundAlpha 格式)

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


def _build_error_response(
    session_id: str,
    error_code: int,
    error_msg: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    构建错误响应

    Args:
        session_id: 会话 ID
        error_code: 错误码
        error_msg: 错误信息
        metadata: 可选的元数据

    Returns:
        JSON 字符串格式的错误响应
    """
    response = {
        "session_id": session_id,
        "error_code": error_code,
        "status_info": error_msg,
        "llm_content": [],
        "metadata": metadata or {},
    }

    return json.dumps(response, ensure_ascii=False)

