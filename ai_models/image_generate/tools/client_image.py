from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import httpx

from config.ai_config import ProviderConfig
from models.utils import (
    retry_operation,
    BaseAPIClient,
    file_url_to_data_uri,
)


# 全局共享 HTTP 客户端连接池（线程安全）
_IMAGE_HTTP_CLIENT: Optional[httpx.Client] = None
_IMAGE_CLIENT_LOCK = threading.Lock()


def _get_image_http_client() -> httpx.Client:
    """获取全局共享的图像生成 HTTP 客户端（线程安全单例）"""
    global _IMAGE_HTTP_CLIENT
    if _IMAGE_HTTP_CLIENT is None:
        with _IMAGE_CLIENT_LOCK:
            if _IMAGE_HTTP_CLIENT is None:
                _IMAGE_HTTP_CLIENT = httpx.Client(
                    timeout=150.0,
                    limits=httpx.Limits(
                        max_connections=20, max_keepalive_connections=10
                    ),
                )
    return _IMAGE_HTTP_CLIENT


class LingyaImageClient(BaseAPIClient):
    """负责与灵芽图片生成/编辑服务交互的客户端。

    根据是否提供参考图片自动选择纯文本生成或编辑接口。
    """

    def __init__(
        self, *, provider: ProviderConfig, model: str, base_url: str | None
    ) -> None:
        super().__init__(provider, base_url)
        self.model = model

        if not self.base_url:
            raise RuntimeError(f"Provider '{provider.name}' 缺少 base_url。")

        self.generation_url = self.base_url

        # 确定编辑接口 URL
        if self.generation_url.endswith("/images/generations"):
            self.edit_url = self.generation_url
        else:
            fallback = self.provider.base_url
            self.edit_url = fallback.rstrip("/") if fallback else self.generation_url

    def generate(
        self,
        *,
        prompt: str,
        resolution: str,
        image_urls: Optional[List[str]] = None,
        image_size: Optional[str] = None,
    ) -> Tuple[str, str]:
        """根据可用素材自动选择生成或编辑模式。返回 (image_url, mime_type)"""
        images_data = self._collect_image_data(image_urls)
        if images_data:
            return self._generate_with_images(prompt=prompt, images=images_data)
        return self._generate_from_text(
            prompt=prompt, resolution=resolution, image_size=image_size
        )

    @retry_operation(max_retries=3)
    def _generate_from_text(
        self, *, prompt: str, resolution: str, image_size: Optional[str] = None
    ) -> Tuple[str, str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "response_format": "url",  # 明确要求返回URL而不是base64
            "aspect_ratio": resolution,  # 使用传入的比例参数
        }
        # 仅当模型为 nano-banana-pro 且提供了 image_size 时才添加该参数
        if self.model == "nano-banana-pro" and image_size:
            payload["image_size"] = image_size

        client = _get_image_http_client()
        response = client.post(
            self.generation_url,
            json=payload,
            headers=self.headers,
            timeout=150,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    @retry_operation(max_retries=3)
    def _generate_with_images(
        self, *, prompt: str, images: List[str]
    ) -> Tuple[str, str]:
        url = self.edit_url.rstrip("/")
        if not url.endswith("/images/edits") and not url.endswith(
            "/images/generations"
        ):
            url = f"{url}/images/generations"  # 图生图也用generations接口
        payload = {
            "model": self.model,
            "prompt": prompt,
            "image": images,
            "response_format": "url",  # 明确要求返回URL
            # 图生图时不支持aspect_ratio设置
        }
        client = _get_image_http_client()
        response = client.post(url, json=payload, headers=self.headers, timeout=150)
        response.raise_for_status()
        return self._parse_response(response.json())

    def _parse_response(self, body: Dict[str, Any]) -> Tuple[str, str]:
        """解析API响应，返回 (image_url, mime_type)。

        根据API文档，当指定response_format="url"时，API保证返回URL字段。
        """
        images = body.get("data") or []
        if not images:
            raise RuntimeError("Lingya 服务未返回图像数据。")
        item = images[0]

        # API应该返回URL字段（因为我们指定了response_format="url"）
        if "url" in item:
            mime = item.get("mime_type") or "image/png"
            return item["url"], mime

        # 不应该走到这里，如果走到这里说明API行为异常
        raise RuntimeError(
            "API未按预期返回URL字段。请检查response_format参数是否生效。"
        )

    @staticmethod
    def _collect_image_data(image_urls: Optional[List[str]]) -> List[str]:
        """收集图片数据，支持URL或本地路径，返回可用于API的图片数据列表"""
        if not image_urls:
            return []
        images: List[str] = []
        for source in image_urls:
            if not source:
                continue
            # 如果是 fileid:// URL，解析为真实 URL
            if source.startswith("fileid://"):
                from media_resource import get_media_registry
                file_id = source[9:]  # 提取 "fileid://" 后的部分
                try:
                    resolved_url = get_media_registry().resolve(file_id, timeout=150.0)
                    # 递归处理解析后的 URL（可能是 file:// 或 http://）
                    if resolved_url.startswith(("http://", "https://")):
                        images.append(resolved_url)
                    elif resolved_url.startswith("data:"):
                        images.append(resolved_url)
                    else:
                        data = file_url_to_data_uri(resolved_url)
                        if data:
                            images.append(data)
                except Exception as e:
                    # 记录错误但不中断整个流程
                    import logging
                    logging.getLogger(__name__).error(f"解析 file_id {file_id} 失败: {e}")
                    continue
            # 如果是HTTP(S) URL，直接使用
            elif source.startswith(("http://", "https://")):
                images.append(source)
            # 如果是data URI，直接使用
            elif source.startswith("data:"):
                images.append(source)
            # 如果是本地路径，转换为base64
            else:
                data = file_url_to_data_uri(source)
                if data:
                    images.append(data)
        return images


__all__ = ["LingyaImageClient"]
