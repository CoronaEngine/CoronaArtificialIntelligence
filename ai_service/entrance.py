import functools
import logging
import os

import importlib
import sys
import threading
from typing import Callable

import yaml
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)
from ai_tools.ai_config_collector import ConfigCollector

logger = logging.getLogger(__name__)

class ai_entrance:
    collector = ConfigCollector()
    if_import = False
    _lock = threading.Lock()

    @classmethod
    def reimport(cls):
        with cls._lock:
            if cls.if_import:
                return
            modules_path = os.path.join(project_dir,"ai_modules")

            config_path = os.path.join(project_dir,'ai_service', "module_settings.yaml")
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 解析模块配置
            if 'modules' in config:
                for module_data in config['modules']:
                    if not module_data.get('enabled', False):
                        logger.debug(f"跳过禁用模块: {module_data.get('name', '')}")
                        return
                    module_name = module_data.get('name', '')
                    module_dir = os.path.join(modules_path, module_name)

                    # 尝试导入 configs/settings.py
                    settings_path = os.path.join(module_dir, "configs", "settings.py")
                    if os.path.exists(settings_path):
                        try:
                            module_path = f"ai_modules.{module_name}.configs.settings"
                            importlib.import_module(module_path)
                            logger.info(f"✓ 成功导入配置模块: {module_name}")
                        except Exception as e:
                            logger.error(f"✗ 导入配置模块失败 {module_name}: {e}")

                    # 尝试导入 base.py
                    base_path = os.path.join(module_dir, "base.py")
                    if os.path.exists(base_path):
                        try:
                            module_path = f"ai_modules.{module_name}.base"
                            importlib.import_module(module_path)
                            logger.info(f"✓ 成功导入基础模块: {module_name}")
                        except Exception as e:
                            logger.error(f"✗ 导入基础模块失败 {module_name}: {e}")

                    # 尝试导入 loader.py
                    loader_path = os.path.join(module_dir, 'tools', "loader.py")
                    if os.path.exists(loader_path):
                        try:
                            module_path = f"ai_modules.{module_name}.tools.loader"
                            importlib.import_module(module_path)
                            logger.info(f"✓ 成功导入loader模块: {module_name}")
                        except Exception as e:
                            logger.error(f"✗ 导入loader模块失败 {module_name}: {e}")
            ai_entrance.if_import = True



def register_entrance(handler_name: str = None):
    """
    将函数注册为 ai_entrance 的静态方法

    Args:
        handler_name: 在 ai_entrance 中的方法名，默认使用原函数名
    """
    def decorator(func: Callable) -> Callable:
        # 确定在 ai_entrance 中的方法名
        method_name = handler_name or func.__name__
        # 创建包装函数
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(ai_entrance, method_name, staticmethod(wrapper))
        return wrapper

    return decorator


def get_ai_entrance():
    from ai_service import entrance
    _AI_ENTRANCE_CLASS = entrance.ai_entrance
    if not _AI_ENTRANCE_CLASS.if_import:
        _AI_ENTRANCE_CLASS.reimport()
    return _AI_ENTRANCE_CLASS
