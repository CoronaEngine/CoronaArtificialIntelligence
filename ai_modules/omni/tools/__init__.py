from __future__ import annotations

from .omni_tools import load_omni_tools
from . import loader  # trigger register_loader so raw dict config becomes OmniModelConfig

__all__ = ["load_omni_tools"]
