from ..config.ai import AIConfig
from typing import Dict, Any
import copy

class ConfigCollector:
    """配置收集器"""

    def __init__(self):
        self._ai_settings = {}
        self._ai_config = AIConfig()
        self._ai_prompts = {}


    def register_setting(self, key: str):
        """装饰器：注册配置函数"""
        def decorator(func):
            result = func()
            self._ai_settings[key] = result
        return decorator

    def register_prompts(self, key: str):
        """装饰器：注册配置函数"""
        def decorator(func):
            result = func()
            self._ai_prompts[key] = result
        return decorator

    def register_loader(self, key: str):
        def decorator(func):
            if key in self._ai_settings:
                result = func(self._ai_settings[key])
                setattr(self._ai_config, key, result)
        return decorator

    @property
    def AI_SETTINGS(self) -> Dict[str, Any]:
        return copy.deepcopy(self._ai_settings)

    @property
    def AIConfig(self) -> AIConfig:
        return copy.deepcopy(self._ai_config)

