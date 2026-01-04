from __future__ import annotations

from typing import List

from langchain_core.tools import BaseTool

from ai_config.ai_config import AIConfig
from ai_modules.mcp.tools.scene_tools import load_scene_tools



def load_mcp_tools(config: AIConfig) -> list[BaseTool]:
    return _load_internal_scene_tools()


def _load_internal_scene_tools() -> List[BaseTool]:
    try:
        from Backend.utils import get_scene_service
        scene_service = get_scene_service()
    except Exception:
        return []
    return load_scene_tools(scene_service)


__all__ = ["load_mcp_tools"]
