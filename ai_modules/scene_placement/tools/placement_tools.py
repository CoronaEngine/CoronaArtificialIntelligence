from __future__ import annotations

import json
import logging
import os
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ai_config.ai_config import AIConfig  # type: ignore
from ai_media_resource import get_media_registry  # type: ignore
from ai_tools.response_adapter import build_part, build_success_result, build_error_result  # type: ignore

from ai_modules.scene_placement.configs.loader import _load_scene_placement_config

logger = logging.getLogger(__name__)

# === 观测点 1：模块是否被 import ===
logger.info("[scene_placement] placement_tools imported, file=%s", __file__)


# -------------------------
# helpers
# -------------------------

def _norm_path(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def _safe_filename(name: str) -> str:
    name = (name or "").strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r'[:*?"<>|]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "download.bin"


def _filename_from_url(url: str) -> str:
    base = os.path.basename((url or "").split("?")[0].rstrip("/")) or "download.bin"
    return _safe_filename(base)


def _download_url(url: str, out_path: Path, *, timeout: float, retries: int, resume: bool) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part = out_path.with_suffix(out_path.suffix + ".part")

    headers: Dict[str, str] = {}
    mode = "wb"
    if resume and part.exists():
        start = part.stat().st_size
        if start > 0:
            headers["Range"] = f"bytes={start}-"
            mode = "ab"

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=timeout, headers=headers) as r:
                r.raise_for_status()
                with open(part, mode) as f:
                    for chunk in r.iter_bytes():
                        if chunk:
                            f.write(chunk)
            part.replace(out_path)
            return out_path
        except Exception as e:
            last_err = e
            logger.warning("[scene_placement] download failed attempt=%s url=%s err=%s", attempt + 1, url, e)

    raise RuntimeError(f"download failed: {url}; last_err={last_err}")


def _maybe_unzip(path: Path, unzip: bool) -> Path:
    if not unzip:
        return path
    if path.is_file() and path.suffix.lower() == ".zip":
        out_dir = path.parent
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(out_dir)
        return out_dir
    return path


def _pick_model_file_from_dir(d: Path) -> Optional[Path]:
    exts = (".obj", ".dae", ".glb", ".gltf", ".fbx")
    for ext in exts:
        cands = list(d.rglob(f"*{ext}"))
        if cands:
            return cands[0]
    any_files = [p for p in d.rglob("*") if p.is_file()]
    return any_files[0] if any_files else None


def _load_template_scene(template_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not template_path:
        return None
    p = Path(template_path)
    if not p.exists():
        logger.warning("[scene_placement] template_scene_path not found: %s", template_path)
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception as e:
        logger.warning("[scene_placement] template_scene_path invalid json: %s err=%s", template_path, e)
        return None


def _new_scene_from_template(template: Optional[Dict[str, Any]], *, scene_name: str, sun_dir: List[float]) -> Dict[str, Any]:
    # 强制输出顶层三字段：name/sun_direction/actors
    name = template.get("name") if (template and isinstance(template.get("name"), str)) else scene_name

    if template and isinstance(template.get("sun_direction"), list) and len(template["sun_direction"]) == 3:
        try:
            sd = [float(template["sun_direction"][0]), float(template["sun_direction"][1]), float(template["sun_direction"][2])]
        except Exception:
            sd = sun_dir
    else:
        sd = sun_dir

    return {"name": name, "sun_direction": sd, "actors": []}


def _save_scene(scene: Dict[str, Any], scene_path: Path) -> None:
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(json.dumps(scene, ensure_ascii=False, indent=2), encoding="utf-8")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _deterministic_layout(room_size: List[float], count: int, *, margin: float, row_z: float) -> List[Dict[str, Any]]:
    if not room_size or len(room_size) < 3:
        room_size = [5.0, 3.0, 5.0]
    L, W, _H = float(room_size[0]), float(room_size[1]), float(room_size[2])

    x_lo = -L / 2.0 + margin
    x_hi = L / 2.0 - margin
    if x_hi < x_lo:
        x_lo, x_hi = -L / 2.0, L / 2.0

    z_lo = -W / 2.0 + margin
    z_hi = W / 2.0 - margin
    z = _clamp(float(row_z), z_lo, z_hi) if z_hi >= z_lo else 0.0

    if count <= 0:
        return []

    if count == 1:
        xs = [(x_lo + x_hi) / 2.0]
    else:
        step = (x_hi - x_lo) / float(count - 1)
        xs = [x_lo + i * step for i in range(count)]

    out: List[Dict[str, Any]] = []
    for i in range(count):
        out.append({"pos": [float(xs[i]), 0.0, float(z)], "rot": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]})
    return out


def _ensure_vec3(x: Any, default: List[float]) -> List[float]:
    if isinstance(x, list) and len(x) == 3:
        try:
            return [float(x[0]), float(x[1]), float(x[2])]
        except Exception:
            return default
    return default


# -------------------------
# schemas
# -------------------------

class PlacementItem(BaseModel):
    object_id: str = Field(..., description="Object identifier")
    name: Optional[str] = Field(None, description="Name")
    mesh_url: Optional[str] = Field(None, description="Remote mesh URL or fileid://...")
    file_name: Optional[str] = Field(None, description="Preferred file name with extension")
    local_path: Optional[str] = Field(None, description="Existing local model file path (preferred if exists)")

    # 可选：上游若提供布局，覆盖规则布局
    pos: Optional[List[float]] = Field(None, description="override pos [x,y,z]")
    rot: Optional[List[float]] = Field(None, description="override rot [x,y,z]")
    scale: Optional[List[float]] = Field(None, description="override scale [x,y,z]")


class DownloadModelInput(BaseModel):
    object_id: str = Field(..., description="Object identifier")
    mesh_url: str = Field(..., description="Mesh URL or fileid://...")
    file_name: Optional[str] = Field(None, description="Preferred file name")
    subdir: Optional[str] = Field(None, description="Optional subdir override")


class PlaceSceneInput(BaseModel):
    scene_path: str = Field(..., description="Output scene.json path")
    scene_name: str = Field("scene", description="Scene name")
    room_size: List[float] = Field(default_factory=lambda: [5, 3, 5], description="Room size (L,W,H)")
    items: List[PlacementItem] = Field(default_factory=list, description="Items")


class UpdateActorTransformInput(BaseModel):
    scene_path: str = Field(..., description="scene.json path")
    actor_name: str = Field(..., description="actor name or source_name")
    pos: Optional[List[float]] = Field(None, description="new pos")
    rot: Optional[List[float]] = Field(None, description="new rot")
    scale: Optional[List[float]] = Field(None, description="new scale")


# -------------------------
# tool loader
# -------------------------

def load_placement_tools(config: AIConfig) -> List[StructuredTool]:
    # === 观测点 2：工具是否被装载到工具池 ===
    logger.info("[scene_placement] load_placement_tools called")

    # 由 register_loader 注入 config.scene_placement；若不存在就从 settings 解析
    cfg = getattr(config, "scene_placement", None)
    if cfg is None:
        raw = getattr(config, "settings", {}).get("scene_placement") if hasattr(config, "settings") else None
        cfg = _load_scene_placement_config(raw)

    template = _load_template_scene(getattr(cfg, "template_scene_path", None))
    sun_dir = [float(cfg.sun_direction_x), float(cfg.sun_direction_y), float(cfg.sun_direction_z)]

    media_registry = get_media_registry()

    def _resolve_mesh_url(mesh_url: str) -> str:
        u = (mesh_url or "").strip()
        if u.startswith("fileid://"):
            media = media_registry.get_by_file_id(u[9:].strip())
            return str(getattr(media, "content_url", "") or "")
        return u

    def download_model_asset(object_id: str, mesh_url: str, file_name: Optional[str] = None, subdir: Optional[str] = None) -> str:
        try:
            url = _resolve_mesh_url(mesh_url)
            if not url:
                raise ValueError("mesh_url 为空或无法解析 fileid://")

            out_dir = Path(str(cfg.asset_root)) / (subdir or str(cfg.model_subdir)) / str(object_id)
            fname = file_name or _filename_from_url(url)
            saved = _download_url(
                url,
                out_dir / fname,
                timeout=float(cfg.request_timeout),
                retries=int(cfg.download_retries),
                resume=bool(cfg.download_resume),
            )
            saved2 = _maybe_unzip(saved, bool(cfg.download_unzip))

            local_path: Path
            if saved2.is_dir():
                pick = _pick_model_file_from_dir(saved2)
                if pick is None:
                    raise RuntimeError(f"解压目录内未找到模型文件: {saved2}")
                local_path = pick
            else:
                local_path = saved2

            part = build_part(
                content_type="file",
                content_text=_norm_path(local_path),
                parameter={"additional_type": ["placement_download"], "object_id": object_id, "name": local_path.name},
            )
            return build_success_result(parts=[part]).to_envelope(interface_type="media")
        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(interface_type="media")

    def place_scene_from_items(scene_path: str, scene_name: str = "scene", room_size: List[float] = [5, 3, 5], items: List[PlacementItem] = []) -> str:
        # === 观测点 3：核心工具是否被调用 ===
        logger.info("[scene_placement] place_scene_from_items called scene_path=%s items=%d", scene_path, len(items or []))
        try:
            sp = Path(scene_path)
            scene = _new_scene_from_template(template, scene_name=scene_name, sun_dir=sun_dir)

            base_layouts = _deterministic_layout(
                room_size=room_size,
                count=len(items or []),
                margin=float(cfg.layout_margin),
                row_z=float(cfg.layout_row_z),
            )

            created: List[Dict[str, Any]] = []
            for idx, it in enumerate(items or []):
                oid = it.object_id
                base = base_layouts[idx] if idx < len(base_layouts) else {"pos": [0, 0, 0], "rot": [0, 0, 0], "scale": [1, 1, 1]}
                pos = _ensure_vec3(it.pos, base["pos"])
                rot = _ensure_vec3(it.rot, base["rot"])
                scale = _ensure_vec3(it.scale, base["scale"])

                # local model path
                local_file: Optional[Path] = None
                if it.local_path:
                    p = Path(it.local_path)
                    if p.exists():
                        local_file = p

                if local_file is None:
                    if not it.mesh_url:
                        raise ValueError(f"object_id={oid} 缺少 mesh_url/local_path")
                    env = json.loads(download_model_asset(oid, it.mesh_url, it.file_name))
                    llm_content = env.get("llm_content", []) or []
                    saved_path = None
                    if llm_content and isinstance(llm_content, list):
                        parts = (llm_content[0] or {}).get("part", []) or []
                        for p in parts:
                            if isinstance(p, dict) and p.get("content_type") == "file":
                                saved_path = p.get("content_text")
                                break
                    if not saved_path:
                        raise RuntimeError(f"object_id={oid} 下载失败：未返回本地路径")
                    local_file = Path(saved_path)

                if local_file is None or not local_file.exists():
                    raise RuntimeError(f"object_id={oid} 本地模型不存在: {local_file}")

                actor_name = it.file_name or local_file.name
                ext = local_file.suffix.lower().lstrip(".")
                actor = {
                    "name": actor_name,
                    "source_name": actor_name,
                    "path": _norm_path(local_file),
                    "type": ext,
                    "geometry": {"pos": pos, "rot": rot, "scale": scale},
                }

                scene["actors"].append(actor)
                created.append(actor)

            _save_scene(scene, sp)

            # === 观测点 4：json 是否真正落盘 ===
            logger.info("[scene_placement] scene.json written: %s (size=%d)", str(sp), sp.stat().st_size)

            scene_text = sp.read_text(encoding="utf-8")
            parts = [
                build_part(
                    content_type="text",
                    content_text=f"✅ 已生成 scene.json\nscene_path: {str(sp)}\nactors: {len(created)}",
                    parameter={"additional_type": ["placement_scene_path"], "scene_path": str(sp), "actors": created},
                ),
                build_part(
                    content_type="file",
                    content_text=scene_text,
                    parameter={"additional_type": ["placement_scene"], "name": "scene.json", "scene_path": str(sp)},
                ),
                build_part(
                    content_type="text",
                    content_text="scene.json 内容如下（可直接复制保存）：\n\n```json\n" + scene_text + "\n```",
                    parameter={"additional_type": ["placement_scene_inline_json"], "scene_path": str(sp)},
                ),
            ]
            return build_success_result(parts=parts).to_envelope(interface_type="media")
        except Exception as e:
            logger.exception("[scene_placement] place_scene_from_items error: %s", e)
            return build_error_result(error_message=str(e)).to_envelope(interface_type="media")

    def update_actor_transform(scene_path: str, actor_name: str, pos: Optional[List[float]] = None, rot: Optional[List[float]] = None, scale: Optional[List[float]] = None) -> str:
        logger.info("[scene_placement] update_actor_transform called scene_path=%s actor_name=%s", scene_path, actor_name)
        try:
            sp = Path(scene_path)
            if not sp.exists():
                raise FileNotFoundError(f"scene.json not found: {sp}")

            scene = json.loads(sp.read_text(encoding="utf-8"))
            actors = scene.get("actors") or []
            if not isinstance(actors, list):
                raise ValueError("scene.actors must be list")

            idx = None
            for i, a in enumerate(actors):
                if isinstance(a, dict) and (a.get("name") == actor_name or a.get("source_name") == actor_name):
                    idx = i
                    break
            if idx is None:
                raise ValueError(f"actor not found: {actor_name}")

            geo = actors[idx].get("geometry") or {}
            if isinstance(pos, list) and len(pos) == 3:
                geo["pos"] = [float(pos[0]), float(pos[1]), float(pos[2])]
            if isinstance(rot, list) and len(rot) == 3:
                geo["rot"] = [float(rot[0]), float(rot[1]), float(rot[2])]
            if isinstance(scale, list) and len(scale) == 3:
                geo["scale"] = [float(scale[0]), float(scale[1]), float(scale[2])]

            actors[idx]["geometry"] = geo
            scene["actors"] = actors
            _save_scene(scene, sp)

            part = build_part(
                content_type="text",
                content_text=f"✅ 已更新 actor: {actor_name}",
                parameter={"additional_type": ["placement_transform_update"], "scene_path": str(sp), "actor_name": actor_name},
            )
            return build_success_result(parts=[part]).to_envelope(interface_type="media")
        except Exception as e:
            logger.exception("[scene_placement] update_actor_transform error: %s", e)
            return build_error_result(error_message=str(e)).to_envelope(interface_type="media")

    return [
        StructuredTool(
            name="download_model_asset",
            description="下载模型文件到本地（支持 mesh_url/fileid://）",
            func=download_model_asset,
            args_schema=DownloadModelInput,
        ),
        StructuredTool(
            name="place_scene_from_items",
            description="按模板格式生成 scene.json（name/sun_direction/actors），并返回 scene.json",
            func=place_scene_from_items,
            args_schema=PlaceSceneInput,
        ),
        StructuredTool(
            name="update_actor_transform",
            description="更新 scene.json 中某个 actor 的变换",
            func=update_actor_transform,
            args_schema=UpdateActorTransformInput,
        ),
    ]


__all__ = ["load_placement_tools"]
