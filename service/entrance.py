import functools
import logging
import os

import importlib
from typing import Dict, Callable
logger = logging.getLogger(__name__)

class ai_entrance:
    def __init__(self):
        ai_models_path = "../ai_models"
        for item in os.listdir(ai_models_path):
            # 检查是否是文件夹且包含 base.py
            item_path = os.path.join(ai_models_path, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "base.py")):
                try:
                    module_path = f"ai_models.{item}.base"

                    logger.debug(f"尝试导入模块: {module_path}")

                    # 导入模块
                    module = importlib.import_module(module_path)
                except Exception as exc:
                    print(f"Warning: {item} not exists in ai_models.{item}, overwriting...")
                    logger.error(f"模块导入异常: {exc}")


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

        # 动态添加到 ai_entrance 类
        if hasattr(ai_entrance, method_name):
            # 如果已存在，可以选择覆盖或跳过
            print(f"Warning: {method_name} already exists in ai_entrance, overwriting...")

        setattr(ai_entrance, method_name, staticmethod(wrapper))

        return wrapper


    return decorator

print(ai_entrance().handle_image_generation(payload = {
        "session_id": "test_session",
        "llm_content": [
            {
                "role": "user",
                "interface_type": "image",
                "part": [{"content_type": "text", "content_text": "一只可爱的小猫咪"}],
            }
        ]
    }))