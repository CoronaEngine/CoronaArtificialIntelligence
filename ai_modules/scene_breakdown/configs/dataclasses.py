from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SceneBreakdownConfig:
    """
    scene_breakdown 模块配置
    """
    enable: bool = True

    # LLM
    temperature: float = 0.6
    request_timeout: float = 60.0

    # 默认参数
    default_style: str = "现代"
    default_detail_level: str = "中等"

    # 3D（下游）默认参数：本模块只负责透传/产出任务 payload
    default_mesh_format: str = "glb"
    default_texture_size: int = 2048
