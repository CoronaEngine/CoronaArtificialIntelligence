from __future__ import annotations

"""ai_modules.scene_placement

- 目标：把已生成/已下载的 3D 模型文件写入 scene.json（对齐模板格式：name/sun_direction/actors）
- 不依赖 ai_models.factory（不走 LLM），使用确定性规则布局
- 通过 import side-effect 强制触发 settings/loader 注册
"""

# 强制触发注册：避免“装饰器没执行导致模块没被加载”
from .configs import setting as setting  # noqa: F401
from .configs import loader as loader  # noqa: F401

from .tools.placement_tools import load_placement_tools

__all__ = ["load_placement_tools"]
