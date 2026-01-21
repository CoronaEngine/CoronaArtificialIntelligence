from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_config.ai_config import AIConfig
from ai_media_resource import get_media_registry
from ai_models.base_pool import get_chat_model
from ai_modules.image_generate.tools.image_tools import load_image_tools
from ai_tools.context import get_current_session
from ai_tools.response_adapter import (
    build_error_result,
    build_part,
    build_success_result,
    resolve_parts,
)


# =========================
# Input Schemas
# =========================


class GenerateObjectListInput(BaseModel):
    scene_type: str = Field(..., description="场景类型，例如：卧室/客厅/厨房")
    style: str = Field(default="现代", description="风格，例如：现代/北欧/日式")
    detail_level: str = Field(default="中等", description="细节程度：简短/中等/详细")
    constraints: Optional[str] = Field(default=None, description="可选约束：小户型/带飘窗等")
    max_objects: int = Field(default=12, description="最多输出多少个物品")


class GenerateImagesFromSelectedInput(BaseModel):
    # 允许不传 selected_ids：前端可能只回传 items 并在每项上标记 selected
    selected_ids: Optional[List[str]] = Field(
        default=None,
        description="用户勾选的物品ID列表（来自 generate_object_list 的 items[].id）。不传则使用 items[].selected 或 items[].selected_default。",
    )
    items: List[Dict[str, Any]] = Field(..., description="generate_object_list 返回的 parameter.items 原样回传")

    style: str = Field(default="现代", description="风格")
    resolution: str = Field(default="1:1", description="画幅比例")
    image_size: str = Field(default="2K", description="图片尺寸")
    max_objects: int = Field(default=12, description="最多生成多少张图（防止一次勾选过多）")


class Generate3DFromSelectedInput(BaseModel):
    # 直接复用上一工具的 items（其中应包含 file_id/url）
    selected_ids: Optional[List[str]] = Field(
        default=None,
        description="用户勾选的物品ID列表（来自 object_images 的 items[].id）。不传则使用 items[].selected。",
    )
    items: List[Dict[str, Any]] = Field(
        ...,
        description="generate_images_from_selected 返回的 parameter.items 原样回传（需包含 file_id 或 fileid 或 url）。",
    )
    mesh_format: str = Field(default="glb", description="输出 3D 格式，例如 glb/obj/fbx（取决于你的 3D 服务支持）")
    max_objects: int = Field(default=12, description="最多生成多少个 3D（防止一次勾选过多）")


# =========================
# Helpers
# =========================


def _extract_json(raw: str) -> dict:
    m = re.search(r"```json\s*(\{.*?\})\s*```", raw, flags=re.S | re.I)
    if not m:
        raise ValueError("模型未输出 JSON（缺少 ```json ... ```）")
    return json.loads(m.group(1))


def _split_multi_object_name(name: str) -> List[str]:
    """把可能出现的多物体 name 拆成单物体（兜底）。"""
    s = (name or "").strip()
    if not s:
        return []
    # 常见连接符：、 ， / + 和 与 以及
    seps = r"[、,/+\n]|\\b(?:和|与|以及)\\b"
    parts = [p.strip(" ,，。、/+") for p in re.split(seps, s) if p.strip(" ,，。、/+")]
    # 去重但保持顺序
    seen = set()
    out = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out[:3]  # 防止模型一次塞太多


def _single_object_prompt(name: str, style: str, desc: str = "") -> str:
    """
    单体物品图强约束：
    - 强行 SINGLE OBJECT ONLY
    - 强负向：scene/room/interior/layout/multi-view
    """
    return f"""
SINGLE OBJECT ONLY: {name}
Generate ONLY ONE object: {name}.

Requirements:
- Object: {name}
- Style: {style}
- View: front view ONLY
- Centered, full object visible, no cropping
- Plain neutral background (white or light gray)
- Studio product lighting, sharp focus
- Realistic proportions

Optional detail: {desc}

NEGATIVE:
- No room, no scene, no interior, no bedroom, no living room
- No layout, no floorplan, no top view, no angle view, no multi-view
- No other objects, no people
- No text, no logo, no watermark
""".strip()


def _pick_selected_items(items: List[Dict[str, Any]], selected_ids: Optional[List[str]], max_objects: int) -> List[Dict[str, Any]]:
    item_map: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if isinstance(it, dict):
            _id = str(it.get("id", "") or "").strip()
            if _id:
                item_map[_id] = it

    selected: List[Dict[str, Any]] = []
    if selected_ids:
        for _id in selected_ids:
            _id = str(_id or "").strip()
            if _id in item_map:
                selected.append(item_map[_id])
    else:
        # 支持 items[].selected / items[].selected_default
        for it in items:
            if not isinstance(it, dict):
                continue
            if it.get("selected") is True or it.get("selected_default") is True:
                selected.append(it)

    max_objects = max(1, int(max_objects))
    return selected[:max_objects]


def _build_image_part(name: str, file_id: str, resolution: str, image_size: str) -> Dict[str, Any]:
    return build_part(
        content_type="image",
        content_text=name,
        file_id=file_id,
        parameter={
            "name": name,
            "resolution": resolution,
            "image_size": image_size,
        },
    )


def _resolve_fileid_to_http(file_id_or_url: str) -> str:
    """把 fileid://xxx 或 file_id 解析成可给上游(3D服务)使用的 http(s)/data URL。"""
    s = (file_id_or_url or "").strip()
    if not s:
        raise ValueError("空的 file_id/url")

    if s.startswith("fileid://"):
        fid = s[len("fileid://") :].lstrip("/")
    else:
        fid = s

    # 3D 上游一般要求 http(s) URL；如果 registry 存的是 cache://，需要返回原始 URL
    url = get_media_registry().resolve(fid, timeout=150.0, return_original_url=True)
    return url


def _call_3d_service(image_url: str, mesh_format: str) -> Tuple[str, Optional[str]]:
    """调用外部 3D 服务。

    默认读环境变量 MESH_SERVICE_URL
    POST JSON: {"image_url": "...", "format": "glb"}
    返回 JSON 支持字段：mesh_url / url / data.url

    你如果已有 3D SDK/接口，把这里替换掉即可。
    """
    base_url = (os.getenv("MESH_SERVICE_URL") or "").strip()
    if not base_url:
        raise RuntimeError(
            "未配置 3D 服务地址。请设置环境变量 MESH_SERVICE_URL，例如：http://127.0.0.1:8000/api/mesh"
        )

    payload = {"image_url": image_url, "format": mesh_format}
    with httpx.Client(timeout=300.0) as client:
        resp = client.post(base_url, json=payload)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}

    mesh_url = data.get("mesh_url") or data.get("url")
    if not mesh_url and isinstance(data.get("data"), dict):
        mesh_url = data["data"].get("url") or data["data"].get("mesh_url")

    if not mesh_url:
        raise RuntimeError(f"3D 服务返回缺少 mesh_url/url 字段：{data}")

    mime_type = data.get("mime_type")
    return str(mesh_url), (str(mime_type) if mime_type else None)


# =========================
# Tool Loader
# =========================


def load_scene_breakdown_tools(config: AIConfig) -> List[StructuredTool]:
    # 文本模型：负责拆解清单
    llm = get_chat_model(category="text", temperature=0.6, request_timeout=60.0)

    # 图像工具：负责生成单体物品图
    image_tools = load_image_tools(config)
    gen_tool: Optional[StructuredTool] = None
    for t in image_tools:
        if getattr(t, "name", "") == "generate_image":
            gen_tool = t
            break

    def _gen_image_file_id(
        session_id: str,
        prompt: str,
        resolution: str,
        image_size: str,
    ) -> str:
        if gen_tool is None:
            raise RuntimeError("generate_image 工具不可用：请检查 image_generate 是否启用、账号池是否配置")

        envelope = gen_tool.invoke(
            {"prompt": prompt, "resolution": resolution, "image_urls": None, "image_size": image_size},
            config={"configurable": {"session_id": session_id}},
        )
        data = json.loads(envelope)
        parts = (data.get("result") or {}).get("parts") or []
        for p in parts:
            if isinstance(p, dict) and p.get("content_type") == "image" and p.get("file_id"):
                return str(p["file_id"])
        raise RuntimeError("generate_image 未返回 file_id")

    # -------------------------
    # Tool A: 生成物品清单
    # -------------------------
    def generate_object_list(
        scene_type: str,
        style: str = "现代",
        detail_level: str = "中等",
        constraints: Optional[str] = None,
        max_objects: int = 12,
    ) -> str:
        try:
            scene_type = (scene_type or "").strip()
            if not scene_type:
                raise ValueError("scene_type 不能为空")

            style = (style or "现代").strip()
            detail_level = (detail_level or "中等").strip()
            constraints = (constraints or "无").strip()
            max_objects = max(1, int(max_objects))

            prompt = f"""
你只输出 JSON，不输出任何自然语言段落。

输入：
- 场景类型：{scene_type}
- 风格：{style}
- 细节程度：{detail_level}
- 约束：{constraints}

硬性规则：
- objects 只包含“具体可单独生成的物体”（例如：现代简约双人床、白色三抽床头柜）
- 每个 name 必须只描述一个物体，不能用“、/和/与/+”连接多个物体
- 不要出现任何：整体/俯视/斜视/鸟瞰/布局/动线/平面图/场景图/参考图
- 不要输出 URL、不要输出图片
- objects 数量 <= {max_objects}

输出格式（严格一致）：
```json
{{
  "objects": [
    {{"name":"物体名称(必须具体,单物体)","desc":"一句话用途/特征"}}
  ]
}}
""".strip()
            resp = llm.invoke(
                [
                    SystemMessage(content="你是室内物品拆解助手，只输出JSON。"),
                    HumanMessage(content=prompt),
                ]
            )

            structured = _extract_json(resp.content or "")
            objects = structured.get("objects", [])
            if not isinstance(objects, list) or not objects:
                raise ValueError("objects 为空或格式不正确")

            items: List[Dict[str, Any]] = []
            for o in objects[:max_objects]:
                if not isinstance(o, dict):
                    continue
                raw_name = str(o.get("name", "") or "").strip()
                if not raw_name:
                    continue
                desc = str(o.get("desc", "") or "").strip()

                # 兜底：把多物体名称拆开
                for name in _split_multi_object_name(raw_name):
                    items.append(
                        {
                            "id": f"obj_{uuid.uuid4().hex[:8]}",
                            "name": name,
                            "desc": desc,
                            "selected_default": True,
                        }
                    )
                    if len(items) >= max_objects:
                        break
                if len(items) >= max_objects:
                    break

            if not items:
                raise ValueError("未生成有效物品清单")

            lines = ["【物品清单（可勾选）】"]
            for it in items:
                d = it.get("desc") or ""
                lines.append(f"- {it['id']} | {it['name']}：{d}".rstrip("："))

            part = build_part(
                content_type="text",
                content_text="\n".join(lines),
                parameter={
                    "additional_type": ["object_list"],
                    "items": items,
                    "final_tool_output": True,
                    "suppress_postprocess": True,
                },
            )
            return build_success_result(parts=[part]).to_envelope(interface_type="text")

        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(interface_type="text")

    # -------------------------
    # Tool B: 清单 -> 逐个生成单体物品图（每个物品一张 image part）
    # -------------------------
    def generate_images_from_selected(
        selected_ids: Optional[List[str]],
        items: List[Dict[str, Any]],
        config: RunnableConfig,
        style: str = "现代",
        resolution: str = "1:1",
        image_size: str = "2K",
        max_objects: int = 12,
    ) -> str:
        try:
            if not items or not isinstance(items, list):
                raise ValueError("items 不能为空（请传入 generate_object_list 返回的 parameter.items）")

            selected = _pick_selected_items(items, selected_ids, max_objects)
            if not selected:
                raise ValueError("未找到任何选中物品（请传 selected_ids 或在 items 内标记 selected）")

            session_id = (config.get("configurable", {}) or {}).get("session_id") or get_current_session()

            # 逐个生成，保证“一物一图”
            results: List[Dict[str, Any]] = []
            image_parts: List[Dict[str, Any]] = []

            for it in selected:
                name = str(it.get("name", "") or "").strip()
                desc = str(it.get("desc", "") or "").strip()
                if not name:
                    continue

                prompt = _single_object_prompt(name=name, style=style, desc=desc)
                file_id = _gen_image_file_id(session_id, prompt, resolution, image_size)

                # 先返回 file_id，URL 交给 resolve_parts 统一解析
                image_parts.append(_build_image_part(name=name, file_id=file_id, resolution=resolution, image_size=image_size))

                results.append(
                    {
                        "id": it.get("id"),
                        "name": name,
                        "desc": desc,
                        "file_id": file_id,
                        "fileid": f"fileid://{file_id}",
                        # url 会在 resolve_parts 后补上
                    }
                )

            # 解析 image_parts 的真实 URL（给 UI 直接展示）
            resolved_image_parts = resolve_parts(image_parts, timeout=150.0)

            # 把解析后的 url 回填到 results（按顺序对齐）
            for r, p in zip(results, resolved_image_parts):
                url = p.get("content_url")
                if url:
                    r["url"] = url

            # summary text part（可选）
            lines = ["【单体物品图（每个物品一张）】"]
            for r in results:
                url = r.get("url") or r.get("fileid")
                lines.append(f"- {r['id']} | {r['name']}：{url}")

            summary_part = build_part(
                content_type="text",
                content_text="\n".join(lines),
                parameter={
                    "additional_type": ["object_images"],
                    "items": results,
                    "final_tool_output": True,
                    "suppress_postprocess": True,
                },
            )

            # ✅ 返回多个 image parts，让前端一张张展示
            all_parts = [summary_part] + resolved_image_parts
            return build_success_result(parts=all_parts).to_envelope(interface_type="text")

        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(interface_type="text")

    # -------------------------
    # Tool C: 选中图片 -> 生成 3D 模型
    # -------------------------
    def generate_3d_from_selected(
        selected_ids: Optional[List[str]],
        items: List[Dict[str, Any]],
        config: RunnableConfig,
        mesh_format: str = "glb",
        max_objects: int = 12,
    ) -> str:
        try:
            if not items or not isinstance(items, list):
                raise ValueError("items 不能为空（请传入 generate_images_from_selected 返回的 parameter.items）")

            selected = _pick_selected_items(items, selected_ids, max_objects)
            if not selected:
                raise ValueError("未找到任何选中物品（请传 selected_ids 或在 items 内标记 selected）")

            mesh_format = (mesh_format or "glb").strip().lower()

            session_id = (config.get("configurable", {}) or {}).get("session_id") or get_current_session()
            reg = get_media_registry()

            results: List[Dict[str, Any]] = []
            lines = [f"【3D 模型生成结果（{mesh_format}）】"]

            for it in selected:
                name = str(it.get("name", "") or "").strip() or "unnamed"

                # 输入图片 URL 优先用已解析的 url；否则用 file_id/fileid 解析
                image_url = (it.get("url") or "").strip()
                if not image_url:
                    if it.get("file_id"):
                        image_url = _resolve_fileid_to_http(str(it["file_id"]))
                    elif it.get("fileid"):
                        image_url = _resolve_fileid_to_http(str(it["fileid"]))
                    elif it.get("content_url"):
                        image_url = _resolve_fileid_to_http(str(it["content_url"]))

                if not image_url:
                    raise RuntimeError(f"物品缺少可用图片：{it}")

                mesh_url, mime = _call_3d_service(image_url=image_url, mesh_format=mesh_format)

                # 可选：把 3D 结果也注册到 registry，得到 file_id 便于后续引用
                mesh_file_id = reg.register(
                    session_id=session_id,
                    content_url=mesh_url,
                    resource_type="model",
                    content_text=f"{name}.{mesh_format}",
                    parameter={"mesh_format": mesh_format, "source_image": image_url},
                )

                row = {
                    "id": it.get("id"),
                    "name": name,
                    "image_url": image_url,
                    "mesh_format": mesh_format,
                    "mesh_url": mesh_url,
                    "mesh_file_id": mesh_file_id,
                    "mesh_fileid": f"fileid://{mesh_file_id}",
                }
                if mime:
                    row["mime_type"] = mime

                results.append(row)
                lines.append(f"- {it.get('id')} | {name}：{mesh_url} (fileid://{mesh_file_id})")

            part = build_part(
                content_type="text",
                content_text="\n".join(lines),
                parameter={
                    "additional_type": ["object_meshes"],
                    "items": results,
                    "final_tool_output": True,
                    "suppress_postprocess": True,
                },
            )
            return build_success_result(parts=[part]).to_envelope(interface_type="text")

        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(interface_type="text")

    return [
        StructuredTool(
            name="generate_object_list",
            description="生成物品清单用于用户勾选：不生成图片、不包含布局动线/多视角。",
            func=generate_object_list,
            args_schema=GenerateObjectListInput,
        ),
        StructuredTool(
            name="generate_images_from_selected",
            description="对用户勾选的物品逐个生成单体物品图：一物一图，返回多个 image parts。",
            func=generate_images_from_selected,
            args_schema=GenerateImagesFromSelectedInput,
        ),
        StructuredTool(
            name="generate_3d_from_selected",
            description="从选中物品的图片生成对应 3D 模型：会先 resolve 图片为 http(s) URL，再调用 3D 服务。",
            func=generate_3d_from_selected,
            args_schema=Generate3DFromSelectedInput,
        ),
    ]

all = ["load_scene_breakdown_tools"]
