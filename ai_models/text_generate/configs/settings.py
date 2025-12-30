"""
大语言模型配置 - 默认预设

包含：
- DEFAULT_SYSTEM_PROMPT: 默认系统提示词
- LLM_SETTINGS: LLM 模型配置（聊天模型、工具模型等）

实际的模型配置和系统提示词应放在 InnerAgentWorkflow/ai_config/llm.py 中。
"""

from __future__ import annotations

from typing import Any, Dict

# ===========================================================================
# 默认系统提示词（简单版）
# ===========================================================================

DEFAULT_SYSTEM_PROMPT = """你是一个 AI 助手，可以帮助用户完成各种任务。"""

# ===========================================================================
# LLM 模型配置（默认预设）
# ===========================================================================

LLM_SETTINGS: Dict[str, Any] = {
    "chat": {
        "provider": "example",
        "model": "gpt-4",
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
    "tool_models": {
        "mcp": {
            "provider": "example",
            "model": "gpt-4",
        }
    },
}
