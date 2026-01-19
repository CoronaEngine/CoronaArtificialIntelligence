from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from ai_config.ai_config import AIConfig
from ai_tools.response_adapter import build_part, build_success_result, build_error_result

from ai_modules.three_d_generate.tools.client_3d import Rodin3DClient


class RodinGenerate3DInput(BaseModel):
    """
    Rodin 3D 生成（显式 mode）
    """
    mode: str = Field(default="image_to_3d", description="image_to_3d / text_to_3d")

    images: Optional[List[str]] = Field(
        default=None,
        description="图片输入（URL 或本地路径）。mode=image_to_3d 时必填",
    )

    prompt: Optional[str] = Field(
        default=None,
        description="文本提示词。mode=text_to_3d 时必填",
    )

    condition_mode: str = Field(default="concat")
    tier: str = Field(default="Regular")
    quality: Optional[str] = None
    seed: Optional[int] = None
    geometry_file_format: str = Field(default="glb")
    material: Optional[str] = None
    addons: Optional[str] = None



def load_3d_tools(config: AIConfig) -> List[StructuredTool]:
    
    # 1) 先尝试从 AIConfig.settings 取（如果框架有灌进去的话）
    # print(123,config)
    threed_config = config.rodin3d

    # # 2) 如果取不到（你现在就是这种情况），fallback 到本模块 settings 函数
    # if not isinstance(settings, dict) or not settings:
    #     from ai_modules.three_d_generate.configs.settings import RODIN_3D_SETTINGS
    #     settings = RODIN_3D_SETTINGS() or {}

    base_url = threed_config.base_url.strip()
    api_key = threed_config.api_key.strip()

    if not base_url:
        raise RuntimeError("Rodin base_url 缺失：请在 settings.rodin_3d.base_url 配置")
    if not api_key:
        raise RuntimeError("Rodin api_key 缺失：请在 settings.rodin_3d.api_key 配置")

    # print(123,base_url,api_key),
    client = Rodin3DClient(
        base_url=base_url,
        api_key=api_key,
        timeout=float(threed_config.request_timeout),
    )

    generate_path = threed_config.generate_path
    status_path = threed_config.status_path
    download_path = threed_config.download_path
    poll_interval = threed_config.poll_interval
    poll_timeout = threed_config.poll_timeout

    def _rodin_generate_3d(
        mode: str = "image_to_3d",
        images: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        condition_mode: str = "concat",
        tier: str = "Regular",
        quality: Optional[str] = None,
        seed: Optional[int] = None,
        geometry_file_format: str = "glb",
        material: Optional[str] = None,
        addons: Optional[str] = None,
    ) -> str:
        try:
            mode = (mode or "").strip()
            images = images or None
            prompt = prompt.strip() if isinstance(prompt, str) else None
            print(images)
            if mode not in {"image_to_3d", "text_to_3d"}:
                raise ValueError("mode 必须是 image_to_3d 或 text_to_3d")

            if mode == "image_to_3d" and not images:
                raise ValueError("image_to_3d 模式必须提供 images")
            if mode == "text_to_3d" and not prompt:
                raise ValueError("text_to_3d 模式必须提供 prompt")

            form_fields: Dict[str, Any] = {
                "prompt": prompt,
                "condition_mode": condition_mode,
                "tier": tier,
                "quality": quality,
                "seed": seed,
                "geometry_file_format": geometry_file_format,
                "material": material,
                "addons": addons,
            }

            result = client.run_to_download_urls(
                generate_path=generate_path,
                status_path=status_path,
                download_path=download_path,
                images=images if mode == "image_to_3d" else None,
                form_fields=form_fields,
                poll_interval=poll_interval,
                poll_timeout=poll_timeout,
            )
            print("Rodin 3D 生成结果：", result)
            parts = []
            for it in result.get("downloads", []):
                url = str(it.get("url", "")).strip()
                if not url:
                    continue
                parts.append(
                    build_part(
                        content_type="file",
                        content_text=url,
                        parameter={
                            "additional_type": ["rodin_3d"],
                            "mode": mode,
                            "geometry_file_format": geometry_file_format,
                            "tier": tier,
                            "quality": quality,
                            "name": it.get("name"),
                        },
                    )
                )

            if not parts:
                raise RuntimeError("Rodin 未返回任何可下载文件")

            return build_success_result(parts=parts).to_envelope(interface_type="media")

        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(interface_type="media")

    return [
        StructuredTool(
            name="rodin_generate_3d",
            description="调用 Rodin API 生成 3D（image_to_3d / text_to_3d）",
            func=_rodin_generate_3d,
            args_schema=RodinGenerate3DInput,
        )
    ]


#