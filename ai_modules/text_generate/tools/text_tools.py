"""
文案与场景生成工具 - 整合多风格规划、空间拆解与画图提示词提取
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage

from ....ai_config.ai_config import AIConfig
from ....ai_models.base_pool import get_chat_model

from ....ai_tools.response_adapter import (
    build_part,
    build_success_result,
    build_error_result,
)


from ..configs.prompts import (
    PRODUCT_TEXT_PROMPTS, 
    MARKETING_TEXT_PROMPTS, 
    CREATIVE_TEXT_PROMPTS,
    PLATFORM_TIPS,
)

# ==========================================
# 1. 定义参数模式 (Schemas)
# ==========================================

class ProductTextInput(BaseModel):
    
    instruction: str = Field(..., description=PRODUCT_TEXT_PROMPTS.fields["instruction"])
    style: str = Field(default="专业", description=PRODUCT_TEXT_PROMPTS.fields["style"])
    length: str = Field(default="中等", description=PRODUCT_TEXT_PROMPTS.fields["length"])
    

class MarketingTextInput(BaseModel):
    
    instruction: str = Field(..., description=MARKETING_TEXT_PROMPTS.fields["instruction"])
    platform: str = Field(default="通用", description=MARKETING_TEXT_PROMPTS.fields["platform"])
    tone: str = Field(default="激励", description=MARKETING_TEXT_PROMPTS.fields["tone"])

class CreativeTextInput(BaseModel):
    
    instruction: str = Field(..., description=CREATIVE_TEXT_PROMPTS.fields["instruction"])
    style: str = Field(default="现代", description=CREATIVE_TEXT_PROMPTS.fields["style"])
    length: str = Field(default="中等", description=CREATIVE_TEXT_PROMPTS.fields["length"])
    

# ==========================================
# 2. 工具加载与执行逻辑
# ==========================================

def load_text_tools(config: AIConfig) -> List[StructuredTool]:
    
    llm = get_chat_model(category="text", temperature=0.8, request_timeout=60.0)

    def _extract_json_block(text: str) -> Tuple[str, Optional[dict]]:
        
        if not text:
            return "", None
        
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
        if m:
            json_str = m.group(1).strip()
            readable = (text[: m.start()] + text[m.end() :]).strip()
            try:
                return readable, json.loads(json_str)
            except Exception:
                return text.strip(), None
            
        m2 = re.search(r"(\{[\s\S]*\})\s*$", text.strip())
        if m2:
            try:
                obj = json.loads(m2.group(1))
                readable = text[: m2.start()].strip()
                return readable, obj
            except Exception:
                pass
            
        return text.strip(), None

    def _process_generation(system_prompt: str, user_prompt: str, text_type: str) -> str:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        try:
            response = llm.invoke(messages)
            part = build_part(
                content_type="text",
                content_text=response.content,
                parameter={"additional_type": [text_type]},
            )
            return build_success_result(parts=[part]).to_envelope(interface_type="text")
        except Exception as e:
            return build_error_result(error_message=str(e)).to_envelope(interface_type="text")

    def _generate_product_text(instruction: str, style: str = "专业", length: str = "中等") -> str:
        length_map = {"简短": "50-80字", "中等": "150-200字", "详细": "300-500字"}
        prompt = PRODUCT_TEXT_PROMPTS.user_prompt.format(
            style=style, length_hint=length_map.get(length, "150-200字"), instruction=instruction
        )
        return _process_generation(PRODUCT_TEXT_PROMPTS.system_prompt, prompt, "product_text")

    def _generate_marketing_text(instruction: str, platform: str = "通用", tone: str = "激励") -> str:
        prompt = MARKETING_TEXT_PROMPTS.user_prompt.format(
            tone=tone, instruction=instruction, platform=platform, platform_tip=PLATFORM_TIPS.get(platform, "")
        )
        return _process_generation(MARKETING_TEXT_PROMPTS.system_prompt, prompt, "marketing_text")
    
    def _generate_creative_text(instruction: str, style: str = "现代", length: str = "中等") -> str:
        length_map = {"简短": "100字以内", "中等": "300-500字", "长篇": "800-1000字"}
        prompt = CREATIVE_TEXT_PROMPTS.user_prompt.format(
            style=style, instruction=instruction, length_hint=length_map.get(length, "300-500字")
        )
        return _process_generation(CREATIVE_TEXT_PROMPTS.system_prompt, prompt, "creative_text")

    # ==========================================
    # 3. 注册 StructuredTool
    # ==========================================
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
