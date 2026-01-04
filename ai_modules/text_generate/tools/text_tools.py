"""
文案生成工具 - 使用豆包 LLM 专职生成各类文案内容
"""

from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage

from ai_config.ai_config import AIConfig
from ai_models import get_chat_model
from ai_modules.text_generate.configs.prompts import PRODUCT_TEXT_PROMPTS, MARKETING_TEXT_PROMPTS, CREATIVE_TEXT_PROMPTS, PLATFORM_TIPS
from ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
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


def load_text_tools(config: AIConfig) -> List[StructuredTool]:
    """
    加载文案生成工具

    该工具使用豆包LLM专门生成各类文案，包括：
    - 产品文案：产品描述、卖点提炼、广告语等
    - 营销文案：活动宣传、社交媒体文案等
    - 创意文案：故事、剧本、诗歌等
    """

    # 检查是否配置了豆包 provider
    if "doubao" not in config.providers:
        print("[警告] 未配置豆包provider，文案生成工具将不可用")
        return []

    # 使用文案专用 LLM（从 TEXT 池或降级到豆包配置）
    # 豆包推荐使用 doubao-pro-32k 或 doubao-lite-32k 等模型
    llm = get_chat_model(
        provider_name="doubao",
        model_name="doubao-1-5-pro-32k-250115",  # 默认使用pro版本，更好的文案质量
        temperature=0.8,  # 较高的温度以增加创意性
        request_timeout=60.0,
        category="text",  # 使用 TEXT 池（文案专用 LLM）
    )

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
    ]

    return tools


__all__ = ["load_text_tools"]
