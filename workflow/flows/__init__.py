"""
Flows 子模块 - 具体工作流实现

每个工作流模块需要导出 WORKFLOWS 字典:
    WORKFLOWS: Dict[int, CompiledStateGraph]

示例:
    # image_scene.py
    WORKFLOWS = {
        10101: scene_generation_graph,
    }

自动发现机制会扫描此目录下所有模块并注册。
"""

__all__: list[str] = []
