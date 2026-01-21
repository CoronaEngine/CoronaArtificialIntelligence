from __future__ import annotations

from typing import List
from langchain_core.tools import StructuredTool

from ai_config.ai_config import AIConfig
from ai_modules.scene_breakdown.tools.scene_breakdown_tools import load_scene_breakdown_tools


def load_tools(config: AIConfig) -> List[StructuredTool]:
    return load_scene_breakdown_tools(config)
