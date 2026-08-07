from __future__ import annotations

from .model_tools import load_hunyuan3d_tools
from . import loader  # 触发 register_loader 将 ai_setting dict 转为 Hunyuan3DSettings

__all__ = ["load_hunyuan3d_tools"]
