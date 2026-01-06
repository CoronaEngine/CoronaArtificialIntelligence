from __future__ import annotations

import json
from typing import Literal, TYPE_CHECKING, Tuple, List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_modules.mcp.configs.prompts import (
    SCENE_QUERY_PROMPTS,
    SCENE_TRANSFORM_PROMPTS,
)
from ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)
DEFAULT_SCENE_NAME = "MainScene"

if TYPE_CHECKING:
    from Backend.utils import SceneApplicationService


class SceneQueryInput(BaseModel):
    scene_name: str = Field(default=DEFAULT_SCENE_NAME, description="要查询的场景名称")
    query: Literal["list_models", "get_model_by_name"] = Field(description="查询类型")
    name: str | None = Field(
        default=None, description=SCENE_QUERY_PROMPTS.fields["model_name"]
    )


class TransformModelInput(BaseModel):
    scene_name: str = Field(default=DEFAULT_SCENE_NAME, description="目标场景名称")
    model_name: str = Field(description="需要变换的模型名称")
    operation: Literal["scale", "move", "rotate"] = Field(
        default="scale", description=SCENE_TRANSFORM_PROMPTS.fields["transform_type"]
    )
    scale_factor: float | None = Field(
        default=None,
        description=SCENE_TRANSFORM_PROMPTS.fields["value"],
    )
    vector: Tuple[float, float, float] | None = Field(
        default=None,
        description=SCENE_TRANSFORM_PROMPTS.fields["axis"],
    )


def _build_scene_query_tool(scene_service: "SceneApplicationService") -> StructuredTool:
    def _query_scene(
        *,
        scene_name: str = DEFAULT_SCENE_NAME,
        query: Literal["list_models", "get_model_by_name"],
        name: str | None = None,
    ) -> str:
        try:
            data = SceneQueryInput(scene_name=scene_name, query=query, name=name)
            scene = scene_service.get_scene(data.scene_name)

            result_data = {}
            if scene is None:
                result_data = {"scene": data.scene_name, "actors": []}
            elif data.query == "list_models":
                actors = [actor.name for actor in scene.get_actors()]
                result_data = {"scene": data.scene_name, "actors": actors}
            elif data.query == "get_model_by_name":
                actor = scene_service._find_actor(scene, data.name or "")
                if actor is None:
                    result_data = {
                        "scene": data.scene_name,
                        "actor": None,
                        "found": False,
                    }
                else:
                    result_data = {
                        "scene": data.scene_name,
                        "actor": actor.name,
                        "path": actor.path,
                        "found": True,
                    }
            else:
                return build_error_result(
                    error_message=f"Unsupported query type: {data.query}"
                ).to_envelope(interface_type="scene")

            # 构建 part
            part = build_part(
                content_type="text",
                content_text=json.dumps(result_data, ensure_ascii=False),
            )

            # 返回成功结果
            return build_success_result(parts=[part]).to_envelope(
                interface_type="scene"
            )
        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(
                interface_type="scene"
            )

    return StructuredTool(
        name="scene_query",
        description=SCENE_QUERY_PROMPTS.tool_description,
        args_schema=SceneQueryInput,
        func=_query_scene,
    )


def _build_transform_tool(scene_service: "SceneApplicationService") -> StructuredTool:
    def _transform_model(
        *,
        scene_name: str = DEFAULT_SCENE_NAME,
        model_name: str,
        operation: Literal["scale", "move", "rotate"] = "scale",
        scale_factor: float | None = None,
        vector: Tuple[float, float, float] | None = None,
    ) -> str:
        try:
            data = TransformModelInput(
                scene_name=scene_name,
                model_name=model_name,
                operation=operation,
                scale_factor=scale_factor,
                vector=vector,
            )
            op = data.operation.lower()
            if op == "scale":
                if data.scale_factor is not None:
                    vector = [data.scale_factor] * 3
                elif data.vector is not None:
                    vector = list(data.vector)
                else:
                    raise ValueError("scale 操作需要提供 scale_factor 或 vector")
                payload = scene_service.apply_transform(
                    data.scene_name, data.model_name, "Scale", vector
                )
            elif op == "move":
                if data.vector is None:
                    raise ValueError("move 操作需要提供 vector")
                payload = scene_service.apply_transform(
                    data.scene_name, data.model_name, "Move", list(data.vector)
                )
            elif op == "rotate":
                if data.vector is None:
                    raise ValueError("rotate 操作需要提供 vector")
                payload = scene_service.apply_transform(
                    data.scene_name, data.model_name, "Rotate", list(data.vector)
                )
            else:
                return build_error_result(
                    error_message=f"Unsupported operation '{data.operation}'"
                ).to_envelope(interface_type="scene")

            # 构建 part
            part = build_part(
                content_type="text",
                content_text=json.dumps(payload, ensure_ascii=False),
            )

            # 返回成功结果
            return build_success_result(parts=[part]).to_envelope(
                interface_type="scene"
            )
        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(
                interface_type="scene"
            )

    return StructuredTool(
        name="transform_model",
        description=SCENE_TRANSFORM_PROMPTS.tool_description,
        args_schema=TransformModelInput,
        func=_transform_model,
    )


def load_scene_tools(scene_service: "SceneApplicationService") -> List[StructuredTool]:
    return [
        _build_scene_query_tool(scene_service),
        _build_transform_tool(scene_service),
    ]


__all__ = ["load_scene_tools"]
