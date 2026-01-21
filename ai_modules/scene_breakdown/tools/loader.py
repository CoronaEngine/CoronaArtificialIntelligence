from __future__ import annotations

from typing import List
from langchain_core.tools import StructuredTool

from ai_config.ai_config import AIConfig
from ai_modules.scene_breakdown.tools.scene_breakdown_tools import load_scene_breakdown_tools


def load_tools(config: AIConfig) -> List[StructuredTool]:
    """
    scene_breakdown 模块工具入口：
    - generate_object_list：生成物品清单（给用户勾选/增删改）
    - generate_images_from_selected：按选中/编辑后的清单逐个生成单体物品图URL
    - generate_3d_from_selected：按选中/编辑后的清单生成单体图 + 3D任务payload（可选尝试调用3D工具）
    """
    return load_scene_breakdown_tools(config)
