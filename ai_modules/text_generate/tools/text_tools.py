"""
文案生成工具 - 使用豆包 LLM 专职生成各类文案内容
"""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage

from ai_config.ai_config import AIConfig
from ai_models.base_pool import get_chat_model
from ai_modules.text_generate.configs.prompts import PRODUCT_TEXT_PROMPTS, MARKETING_TEXT_PROMPTS, CREATIVE_TEXT_PROMPTS, PLATFORM_TIPS
from ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)

from typing import Optional, List

from ai_tools.context import get_current_session
from ai_media_resource import get_media_registry, get_storage_adapter, calculate_expire_time
from ai_models.base_pool import get_pool_registry, MediaCategory, ImageRequest, MediaResult
from ai_modules.text_generate.configs.prompts import SCENE_PLAN_PROMPTS

import json
import re
from typing import List, Optional, Tuple
from ai_modules.text_generate.configs.prompts import (
    PRODUCT_TEXT_PROMPTS, MARKETING_TEXT_PROMPTS, CREATIVE_TEXT_PROMPTS,
    PLATFORM_TIPS, SCENE_BREAKDOWN_PROMPTS, SCENE_PLAN_PROMPTS,
)


# 定义参数模式
class ProductTextInput(BaseModel):
    """产品文案生成的输入参数"""

    instruction: str = Field(..., description=PRODUCT_TEXT_PROMPTS.fields["instruction"])
    style: str = Field(
        default="专业", description=PRODUCT_TEXT_PROMPTS.fields["style"]
    )
    length: str = Field(default="中等", description=PRODUCT_TEXT_PROMPTS.fields["length"])


class MarketingTextInput(BaseModel):
    """营销文案生成的输入参数"""

    instruction: str = Field(..., description=MARKETING_TEXT_PROMPTS.fields["instruction"])
    platform: str = Field(
        default="通用", description=MARKETING_TEXT_PROMPTS.fields["platform"]
    )
    tone: str = Field(
        default="激励", description=MARKETING_TEXT_PROMPTS.fields["tone"]
    )


class CreativeTextInput(BaseModel):
    """创意文案生成的输入参数"""

    instruction: str = Field(..., description=CREATIVE_TEXT_PROMPTS.fields["instruction"])
    style: str = Field(
        default="现代", description=CREATIVE_TEXT_PROMPTS.fields["style"]
    )
    length: str = Field(default="中等", description=CREATIVE_TEXT_PROMPTS.fields["length"])

class SceneBreakdownInput(BaseModel):
    """场景拆解（文本 + 结构化list）"""

    scene_type: str = Field(..., description=SCENE_BREAKDOWN_PROMPTS.fields["scene_type"])
    style: str = Field(default="现代", description=SCENE_BREAKDOWN_PROMPTS.fields["style"])
    detail_level: str = Field(default="中等", description=SCENE_BREAKDOWN_PROMPTS.fields["detail_level"])
    constraints: Optional[str] = Field(default=None, description=SCENE_BREAKDOWN_PROMPTS.fields["constraints"])


class ScenePlanInput(BaseModel):
    """
    场景规划（兼容旧入口）
    重要：现在不会生成多视角图，也不会输出动线/布局逻辑
    """

    scene_type: str = Field(..., description=SCENE_PLAN_PROMPTS.fields["scene_type"])
    style: str = Field(default="现代", description=SCENE_PLAN_PROMPTS.fields["style"])
    detail_level: str = Field(default="中等", description=SCENE_PLAN_PROMPTS.fields["detail_level"])
    constraints: Optional[str] = Field(default=None, description=SCENE_PLAN_PROMPTS.fields["constraints"])

    # 兼容字段：保留不删，避免前端/调用方传参报错，但工具内部不再使用
    views: List[str] = Field(default_factory=lambda: ["overall", "top", "angle"], description=SCENE_PLAN_PROMPTS.fields.get("views", ""))
    image_size: str = Field(default="2K", description=SCENE_PLAN_PROMPTS.fields.get("image_size", ""))
    resolution: str = Field(default="1:1", description=SCENE_PLAN_PROMPTS.fields.get("resolution", ""))

def load_text_tools(config: AIConfig) -> List[StructuredTool]:
    """
    加载文案生成工具

    该工具使用豆包LLM专门生成各类文案，包括：
    - 产品文案：产品描述、卖点提炼、广告语等
    - 营销文案：活动宣传、社交媒体文案等
    - 创意文案：故事、剧本、诗歌等
    """

    # 使用 TEXT 池中的 LLM（账号池自动选择最优账号）
    llm = get_chat_model(
        category="text",  # 从 TEXT 账号池获取
        temperature=0.8,  # 较高的温度以增加创意性
        request_timeout=60.0,
    )


    def _extract_json_block(text: str) -> Tuple[str, Optional[dict]]:
        """
        期望模型输出包含两部分：
        1) 可读文本（你现在那种【1】【2】结构）
        2) ```json ... ``` 结构化 JSON
        返回：(readable_text, json_dict_or_none)
        """
        if not text:
            return "", None

        # 找 ```json ... ```
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
        if m:
            json_str = m.group(1).strip()
            readable = (text[: m.start()] + text[m.end() :]).strip()
            try:
                return readable, json.loads(json_str)
            except Exception:
                return text.strip(), None

        # 如果没 fenced code，尝试找最后一个大括号 JSON
        m2 = re.search(r"(\{[\s\S]*\})\s*$", text.strip())
        if m2:
            try:
                obj = json.loads(m2.group(1))
                readable = text[: m2.start()].strip()
                return readable, obj
            except Exception:
                pass

        return text.strip(), None

    
    def _generate_scene_breakdown(
            scene_type: str,
            style: str = "现代",
            detail_level: str = "中等",
            constraints: Optional[str] = None,
        ) -> str:
            """
            输出：
            - 可读文本：只包含 物体清单 / 空间分区 / 可选增补
            - 结构化 JSON：objects/zones/optional_additions（全部是 list）
            ✅ 不包含 layout_logic（布局逻辑与动线）
            ✅ 不提及多视角参考图
            """
            try:
                constraints = constraints or "无"

                user_prompt = f"""
    请根据以下信息，生成“场景拆解”。必须同时输出两部分：

    A) 可读文本（用小标题与编号，格式清晰）：
    【1 物体清单】每项包含：名称、功能、建议位置
    【2 空间分区】每个分区包含哪些物体
    【3 可选增补】3-6个（提升氛围/实用性）

    B) 结构化 JSON（必须放在 ```json 代码块内```，字段固定为）：
    {{
    "scene_type": "...",
    "style": "...",
    "objects": [{{"name":"", "function":"", "position":""}}, ...],
    "zones": [{{"name":"", "objects":["",""]}}, ...],
    "optional_additions": [{{"name":"", "why":"", "position":""}}, ...]
    }}

    【硬性规则】
    - 禁止输出【布局逻辑与动线】或任何 layout_logic 字段
    - 禁止提及整体/俯视/斜视/多视角参考图/鸟瞰/平面图/布局图
    - 禁止输出图片、链接或 Markdown 图片语法
    - 不要提及任何函数名/工具名/API 调用

    场景类型：{scene_type}
    风格：{style}
    细节程度：{detail_level}
    约束：{constraints}
    """.strip()

                messages = [
                    SystemMessage(content="你是一个专业的室内场景拆解助手，只输出最终结果，不要解释过程，不要输出图片/链接。"),
                    HumanMessage(content=user_prompt),
                ]

                response = llm.invoke(messages)
                raw = response.content or ""
                readable_text, structured = _extract_json_block(raw)

                scene_breakdown = structured if isinstance(structured, dict) else None

                part = build_part(
                    content_type="text",
                    content_text=readable_text,
                    parameter={
                        "additional_type": ["scene_breakdown"],
                        "scene_breakdown": scene_breakdown,  # list 结构
                        "scene_type": scene_type,
                        "style": style,
                        "detail_level": detail_level,

                        # ✅ 减少上游再复述一次
                        "final_tool_output": True,
                        "suppress_postprocess": True,
                    },
                )

                return build_success_result(parts=[part]).to_envelope(interface_type="text")

            except Exception as e:
                return build_error_result(error_message=str(e)).to_envelope(interface_type="text")
        
    def _process_generation(
        system_prompt: str,
        user_prompt: str,
        text_type: str,
    ) -> str:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = llm.invoke(messages)

            # 构建 part
            part = build_part(
                content_type="text",
                content_text=response.content,
                parameter={
                    "additional_type": [text_type],
                },
            )

            # 返回成功结果
            return build_success_result(
                parts=[part],
            ).to_envelope(interface_type="text")
        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(
                interface_type="text"
            )

    def _generate_product_text(
        instruction: str,
        style: str = "专业",
        length: str = "中等",
    ) -> str:
        """
        生成产品文案

        Args:
            instruction: 产品描述及要求
            style: 文案风格（可选：专业、活泼、高端、亲切、幽默）
            length: 文案长度（可选：简短、中等、详细）

        Returns:
            生成的产品文案
        """
        length_map = {
            "简短": "50-80字",
            "中等": "150-200字",
            "详细": "300-500字",
        }

        prompt = PRODUCT_TEXT_PROMPTS.user_prompt.format(
            style=style,
            length_hint=length_map.get(length, "150-200字"),
            instruction=instruction,
        )

        return _process_generation(
            system_prompt=PRODUCT_TEXT_PROMPTS.system_prompt,
            user_prompt=prompt,
            text_type="product_text",
        )

    def _generate_marketing_text(
        instruction: str,
        platform: str = "通用",
        tone: str = "激励",
    ) -> str:
        """
        生成营销文案

        Args:
            instruction: 营销活动描述及要求
            platform: 投放平台（可选：通用、微信、微博、抖音、小红书）
            tone: 文案语气（可选：激励、温暖、紧迫、趣味）

        Returns:
            生成的营销文案
        """
        prompt = MARKETING_TEXT_PROMPTS.user_prompt.format(
            tone=tone,
            instruction=instruction,
            platform=platform,
            platform_tip=PLATFORM_TIPS.get(platform, PLATFORM_TIPS["通用"]),
        )

        return _process_generation(
            system_prompt=MARKETING_TEXT_PROMPTS.system_prompt,
            user_prompt=prompt,
            text_type="marketing_text",
        )
    

    def _generate_creative_text(
        instruction: str,
        style: str = "现代",
        length: str = "中等",
    ) -> str:
        """
        生成创意文案

        Args:
            instruction: 创作主题及要求
            style: 创作风格（可选：现代、古典、浪漫、科技、悬疑等）
            length: 作品长度（可选：简短、中等、长篇）

        Returns:
            生成的创意文案
        """
        length_map = {
            "简短": "100字以内",
            "中等": "300-500字",
            "长篇": "800-1000字",
        }

        prompt = CREATIVE_TEXT_PROMPTS.user_prompt.format(
            style=style,
            instruction=instruction,
            length_hint=length_map.get(length, "300-500字"),
        )

        return _process_generation(
            system_prompt=CREATIVE_TEXT_PROMPTS.system_prompt,
            user_prompt=prompt,
            text_type="creative_text",
        )
        
    def _generate_scene_plan(
        scene_type: str,
        style: str = "现代",
        detail_level: str = "中等",
        constraints: Optional[str] = None,
        views: Optional[List[str]] = None,
        image_size: str = "2K",
        resolution: str = "1:1",
    ) -> str:
        """
        ✅ 兼容旧工具名 generate_scene_plan：
        - 现在不再生成多视角图
        - 现在不再输出动线/布局逻辑
        - 现在不再二次调用 LLM（直接复用 breakdown 的输出）
        """
        return _generate_scene_breakdown(
            scene_type=scene_type,
            style=style,
            detail_level=detail_level,
            constraints=constraints,
        )


    # 创建三个结构化工具，带有明确的参数模式
    tools = [
        StructuredTool(
            name="generate_product_text",
            description=PRODUCT_TEXT_PROMPTS.tool_description,
            func=_generate_product_text,
            args_schema=ProductTextInput,
        ),
        StructuredTool(
            name="generate_marketing_text",
            description=MARKETING_TEXT_PROMPTS.tool_description,
            func=_generate_marketing_text,
            args_schema=MarketingTextInput,
        ),
        StructuredTool(
            name="generate_creative_text",
            description=CREATIVE_TEXT_PROMPTS.tool_description,
            func=_generate_creative_text,
            args_schema=CreativeTextInput,
        ),
        StructuredTool(
            name="generate_scene_breakdown",
            description=SCENE_BREAKDOWN_PROMPTS.tool_description,
            func=_generate_scene_breakdown,
            args_schema=SceneBreakdownInput,
        ),
        StructuredTool(
            name="generate_scene_plan",
            description=SCENE_PLAN_PROMPTS.tool_description,
            func=_generate_scene_plan,
            args_schema=ScenePlanInput,
        ),
    ]

    return tools


__all__ = ["load_text_tools"]
