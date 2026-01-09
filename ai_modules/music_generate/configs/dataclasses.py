from dataclasses import dataclass

@dataclass(frozen=False)
class MusicConfig:
    """音乐生成配置"""

    api_key: str | None = None
    base_url: str | None = None
