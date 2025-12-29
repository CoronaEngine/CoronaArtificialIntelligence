from __future__ import annotations


from typing import Any, Dict
import logging

from config.ai_config import get_ai_config

from service.common import (
    ensure_dict,
    build_error_response,
    build_success_response,
    session_context,
    extract_parameter,
    parse_tool_response,
)
from service.concurrency import session_concurrency

logger = logging.getLogger(__name__)


def _extract_text(request_data: Dict[str, Any]) -> str:
    if "text" in request_data:
        return request_data.get("text", "")
    llm_content = request_data.get("llm_content")
    if isinstance(llm_content, list) and llm_content:
        parts = llm_content[0].get("part", [])
        txt = "\n".join(
            p.get("content_text", "") for p in parts if p.get("content_type") == "text"
        ).strip()
        logger.debug(f"提取到 text: {txt}")
        return txt
    return ""


def handle_speech_generation(payload: Any) -> str:
    """语音生成三层结构。"""
    request_data: Dict[str, Any] = ensure_dict(payload)
    metadata = request_data.get("metadata", {})
    session_id = request_data.get("session_id") or "default"
    cfg = get_ai_config()

    # 使用统一的并发控制
    with session_concurrency(session_id, cfg) as acquired:
        if not acquired:
            return build_error_response(
                interface_type="speech",
                session_id=session_id,
                metadata=metadata,
                exc=RuntimeError("并发繁忙，请稍后重试"),
            )
        return _handle_speech_generation_inner(request_data, session_id, metadata, cfg)


def _handle_speech_generation_inner(
    request_data: Dict[str, Any],
    session_id: str,
    metadata: Dict[str, Any],
    cfg,
) -> str:
    """语音生成内部实现（在并发控制内执行）"""
    try:
        logger.debug(f"收到语音生成请求: {request_data}")
        text = _extract_text(request_data)
        if not text:
            raise ValueError("缺少待合成的文本")

        from tools.media.speech_tools import (
            load_speech_tools,
        )

        tools = load_speech_tools(cfg)
        if not tools:
            raise RuntimeError("TTS语音合成功能未启用或配置不完整")
        speech_tool = tools[0]
        tool_params = {
            "text": text,
            "voice_type": extract_parameter(
                request_data, "voice_type", "zh_female_cancan_mars_bigtts"
            ),
            "speed_ratio": extract_parameter(request_data, "speed_ratio", 1.0),
            "loudness_ratio": extract_parameter(request_data, "loudness_ratio", 1.0),
            "encoding": extract_parameter(request_data, "encoding", "mp3"),
            "rate": extract_parameter(request_data, "rate", 24000),
            "max_wait_seconds": extract_parameter(request_data, "max_wait_seconds", 60),
            "poll_interval": extract_parameter(request_data, "poll_interval", 2.0),
        }
        logger.debug(f"speech_tool 参数: {tool_params}")
        with session_context(session_id) as sid:
            logger.debug(f"进入 session_context: {sid}")
            result_json = speech_tool.func(**tool_params)
            session_id = sid
        logger.debug(f"speech_tool 返回: {result_json}")

        # 解析 Tool 返回的 Envelope JSON
        tool_envelope = parse_tool_response(result_json)
        logger.debug(f"解析 tool_envelope: {tool_envelope}")

        # 检查错误
        if tool_envelope.get("error_code", 0) != 0:
            error_msg = tool_envelope.get("status_info", "未知错误")
            logger.error(f"语音合成失败: {error_msg}")
            raise RuntimeError(f"语音合成失败: {error_msg}")

        # 提取 llm_content
        llm_content = tool_envelope.get("llm_content", [])
        if not llm_content:
            logger.error("语音合成未返回有效内容")
            raise RuntimeError("语音合成未返回有效内容")

        # 提取并清洗 parts
        original_parts = llm_content[0].get("part", [])
        cleaned_parts = []
        for part in original_parts:
            cleaned_part = {
                "content_type": part.get("content_type"),
                "content_url": part.get("content_url"),
                "content_text": part.get("content_text", ""),
            }
            # 保留 url_expire_time 字段
            if "url_expire_time" in part:
                cleaned_part["url_expire_time"] = part["url_expire_time"]
            # 严格过滤 parameter
            if "parameter" in part:
                original_param = part["parameter"]
                cleaned_param = {}
                if "speech_type" in original_param:
                    cleaned_param["speech_type"] = original_param["speech_type"]
                if "duration" in original_param:
                    cleaned_param["duration"] = original_param["duration"]
                if cleaned_param:
                    cleaned_part["parameter"] = cleaned_param

            # 移除 None 值字段
            cleaned_part = {k: v for k, v in cleaned_part.items() if v is not None}
            cleaned_parts.append(cleaned_part)
        logger.debug(f"清洗后的 parts: {cleaned_parts}")

        if not cleaned_parts:
            logger.error("语音合成未返回有效的音频部分")
            raise RuntimeError("语音合成未返回有效的音频部分")

        # 解析 parts 中的 fileid:// URL（返回真实 OSS URL 给用户）
        from tools.response_adapter import (
            resolve_parts,
        )

        try:
            cleaned_parts = resolve_parts(cleaned_parts, timeout=150.0)
            logger.debug(f"解析后的 parts: {cleaned_parts}")
        except Exception as e:
            logger.error(f"解析 parts 中的 file_id 失败: {e}")
            # 解析失败时抛出异常，触发错误响应
            raise RuntimeError(f"语音资源解析失败: {e}") from e

        return build_success_response(
            interface_type="speech",
            session_id=session_id,
            metadata=metadata,
            parts=cleaned_parts,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"语音生成异常: {exc}")
        return build_error_response(
            interface_type="speech",
            session_id=session_id,
            metadata=metadata,
            exc=exc,
        )


__all__ = ["handle_speech_generation"]
