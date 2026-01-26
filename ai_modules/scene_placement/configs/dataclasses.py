from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=False)
class ScenePlacementConfig:
    """场景摆放配置"""

    asset_root: str = "assets"
    model_subdir: str = "model"

    request_timeout: float = 120.0
    download_retries: int = 2
    download_resume: bool = True
    download_unzip: bool = True

    # 规则布局参数（不依赖 LLM）
    layout_margin: float = 0.5
    layout_row_z: float = -1.0

    # 可选：指定一个“模板 scene.json”，生成时严格对齐其字段结构与默认值
    # 例如：与你上传的 场景1.json 同结构（name/sun_direction/actors/geometry...）
    template_scene_path: Optional[str] = None

    # 默认光照方向（当 template 中没有或 template_scene_path 为空时使用）
    sun_direction_x: float = -11.0
    sun_direction_y: float = 1.0
    sun_direction_z: float = 1.0
