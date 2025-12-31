from ....config.ai import AIConfig
from ..configs.dataclasses import MediaToolConfig


def _is_media_tool_enabled(cfg: MediaToolConfig, config: AIConfig) -> bool:
    """检查媒体工具是否启用"""
    if not cfg.enable:
        return False
    if not cfg.provider or not cfg.model:
        return False
    if cfg.provider not in config.providers:
        return False
    provider = config.providers[cfg.provider]
    return bool(provider.api_key)
