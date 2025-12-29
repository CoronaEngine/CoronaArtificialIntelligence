"""
LLM 配置加载器
"""

from typing import Any, Dict, Mapping

from config.dataclasses.llm import ToolModelConfig


def _load_tool_models(raw: Mapping[str, Any]) -> Dict[str, ToolModelConfig]:
    """加载工具模型配置"""
    tool_models: Dict[str, ToolModelConfig] = {}
    for name, cfg in raw.items():
        if not isinstance(cfg, Mapping):
            continue
        provider = cfg.get("provider")
        model = cfg.get("model")
        if not provider or not model:
            continue
        tool_models[name] = ToolModelConfig(provider=str(provider), model=str(model))
    return tool_models
