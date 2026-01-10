"""
请求参数解析、处理工具。
"""

from typing import Any, Dict, List


def normalize_image_size(value: str | None) -> str | None:
    """规范化 image_size 表达，如 '1k' -> '1K'.

    仅进行大小写归一，不更改语义；空值直接返回。
    """
    if value is None:
        return None
    v = value.strip()
    # 常见写法：1k/2k/4k/8k，统一为大写 K
    if v.lower().endswith("k"):
        return v[:-1] + "K"
    return v


def extract_prompt_from_llm_content(data: Dict[str, Any]) -> str:
    """从 llm_content 中提取 prompt 文本。

    Args:
        data: 请求数据字典

    Returns:
        提取的 prompt 字符串
    """
    llm_content = data.get("llm_content")
    if not isinstance(llm_content, list) or not llm_content:
        return ""
    first = llm_content[0]
    parts = first.get("part", [])
    prompt = "".join(
        p.get("content_text", "") for p in parts if p.get("content_type") == "text"
    ).strip()
    return prompt


def extract_images_from_request(request_data: Dict[str, Any]) -> List[str]:
    """从 llm_content 中提取图片 URL 列表。

    规则：
    1. 遍历 llm_content[0]["part"] 中的所有 image/detection 类型 part。
    2. 按顺序收集所有图片 URL。

    Args:
        request_data: 请求数据字典

    Returns:
        图片 URL 列表
    """
    llm_content = request_data.get("llm_content", [])
    if not isinstance(llm_content, list) or not llm_content:
        return []

    parts = llm_content[0].get("part", [])
    image_urls: List[str] = []

    for part in parts:
        content_type = part.get("content_type")
        if content_type not in ("image", "detection"):
            continue
        url = part.get("content_url")
        if url:
            image_urls.append(url)

    return image_urls
