import functools
import os

import importlib
from typing import Dict, Callable


class ai_entrance:
    pass

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
