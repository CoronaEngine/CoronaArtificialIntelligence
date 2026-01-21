"""
文本生成工具提示词配置

包含：
- PRODUCT_TEXT_PROMPTS: 产品文案生成
- MARKETING_TEXT_PROMPTS: 营销文案生成
- CREATIVE_TEXT_PROMPTS: 创意文案生成
- PLATFORM_TIPS: 平台文案建议
- TEXT_TOOL_PROMPTS: 文本工具提示词集合
"""

from __future__ import annotations

from typing import Dict

from ai_config.prompts import TextToolPromptConfig, TextToolPrompts,ToolPromptConfig

# ===========================================================================
# 平台文案建议
# ===========================================================================

PLATFORM_TIPS: Dict[str, str] = {
    "微信": "适合长文，可以加入emoji表情",
    "微博": "140字内，简洁有力",
    "抖音": "口语化，有节奏感",
    "小红书": "种草型，真实分享感",
    "通用": "适中长度，普适性强",
}

# ===========================================================================
# 产品文案生成
# ===========================================================================

PRODUCT_TEXT_PROMPTS = TextToolPromptConfig(
    tool_description="生成产品文案，包括产品描述、卖点提炼、广告语等",
    fields={
        "instruction": "产品描述及要求",
        "style": "文案风格，可选：专业、活泼、高端、亲切、幽默",
        "length": "文案长度，可选：简短、中等、详细",
    },
    system_prompt="你是一位专业的文案撰写专家，擅长创作各类营销文案、产品文案和创意内容。",
    user_prompt='''请根据以下产品描述和要求，生成{style}风格的文案，长度约{length_hint}：

需求描述：{instruction}

要求：
1. 突出产品的核心卖点
2. 语言{style}且吸引人
3. 适合用于产品宣传和推广
4. 直接输出文案内容，不要输出任何解释''',
)

# ===========================================================================
# 营销文案生成
# ===========================================================================

MARKETING_TEXT_PROMPTS = TextToolPromptConfig(
    tool_description="生成营销文案，包括活动宣传、社交媒体文案等",
    fields={
        "instruction": "营销活动描述及要求",
        "platform": "投放平台，可选：通用、微信、微博、抖音、小红书",
        "tone": "文案语气，可选：激励、温暖、紧迫、趣味",
    },
    system_prompt="你是一位专业的营销文案专家，精通各类平台的文案创作和用户心理。",
    user_prompt='''请根据以下营销活动描述，生成{tone}语气的文案：

需求描述：{instruction}
投放平台：{platform}

平台建议：{platform_tip}

要求：
1. 符合目标受众的语言习惯
2. 突出营销要点和优惠信息
3. 语气{tone}，能够引发行动
4. 直接输出文案内容，不要输出任何解释''',
)

# ===========================================================================
# 创意文案生成
# ===========================================================================

CREATIVE_TEXT_PROMPTS = TextToolPromptConfig(
    tool_description="生成创意文案，包括故事、剧本、诗歌等",
    fields={
        "instruction": "创作主题及要求",
        "style": "创作风格，可选：现代、古典、浪漫、科技、悬疑等",
        "length": "作品长度，可选：简短、中等、长篇",
    },
    system_prompt="你是一位富有创意的文案创作者，擅长各种文学体裁和创意表达。",
    user_prompt='''请根据以下主题和要求，创作一个{style}风格的作品：

需求描述：{instruction}
作品长度：{length_hint}

要求：
1. 创意新颖，富有想象力
2. 风格符合{style}特点
3. 内容完整，结构合理
4. 直接输出作品内容，不要输出任何解释''',
)

# ===========================================================================
# 文本工具提示词集合
# ===========================================================================

TEXT_TOOL_PROMPTS = TextToolPrompts(
    product=PRODUCT_TEXT_PROMPTS,
    marketing=MARKETING_TEXT_PROMPTS,
    creative=CREATIVE_TEXT_PROMPTS,
    platform_tips=PLATFORM_TIPS,
)

# ===========================================================================
# 场景拆解提示词集合
# ===========================================================================

SCENE_PLAN_PROMPTS = ToolPromptConfig(
    tool_description=(
        "场景规划工具：输入一个场景类型（如卧室/客厅），输出："
        "① 场景拆解文本（物体清单+空间分区+可选增补）。"
        "本工具不生成参考图，不输出动线/布局逻辑。"
    ),
    fields={
        "scene_type": "场景类型，例如：卧室、客厅、厨房、办公室等",
        "style": "风格，例如：现代、北欧、日式、极简等",
        "detail_level": "细节程度：简短/中等/详细",
        "constraints": "可选：约束，例如：小户型、主卧、儿童房、带飘窗等",
        # 注意：views/image_size/resolution 字段仍可保留兼容，但工具内部将不再使用它们
        "views": "（兼容字段）参考视角列表（已不再使用）",
        "image_size": "（兼容字段）生成图片尺寸等级（已不再使用）",
        "resolution": "（兼容字段）画幅比例（已不再使用）",
    },
)

SCENE_BREAKDOWN_PROMPTS = ToolPromptConfig(
    tool_description="生成场景拆解：输出可读文本 + 结构化JSON（objects/zones/optional_additions）。不输出动线/布局逻辑，不生成图片。",
    fields={
        "scene_type": "场景类型，例如：办公室、卧室、客厅等",
        "style": "风格，例如：现代、北欧、日式、极简等",
        "detail_level": "细节程度：简短/中等/详细",
        "constraints": "可选约束：小户型/主卧/带飘窗等",
    },
)

# ===========================================================================
# 导出
# ===========================================================================

__all__ = [
    "PRODUCT_TEXT_PROMPTS",
    "MARKETING_TEXT_PROMPTS",
    "CREATIVE_TEXT_PROMPTS",
    "TEXT_TOOL_PROMPTS",
    "PLATFORM_TIPS",
    "SCENE_BREAKDOWN_PROMPTS",
    "SCENE_PLAN_PROMPTS",
]
