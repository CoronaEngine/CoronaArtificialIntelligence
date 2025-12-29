"""
网络配置加载器
"""

from typing import Any, Mapping

from config.dataclasses.network import (
    NetworkConfig,
    PollingConfig,
)


def _load_network_config(raw: Mapping[str, Any] | None) -> NetworkConfig:
    """加载网络配置"""
    if not isinstance(raw, Mapping):
        return NetworkConfig()
    return NetworkConfig(
        request_timeout=int(raw.get("request_timeout", 60)),
        download_timeout=int(raw.get("download_timeout", 300)),
        download_chunk_size=int(raw.get("download_chunk_size", 8192)),
        download_retries=int(raw.get("download_retries", 2)),
        download_backoff_factor=float(raw.get("download_backoff_factor", 0.5)),
    )


def _load_polling_config(raw: Mapping[str, Any] | None) -> PollingConfig:
    """加载轮询配置"""
    if not isinstance(raw, Mapping):
        return PollingConfig()

    service_intervals = raw.get("service_intervals", {})
    if not isinstance(service_intervals, Mapping):
        service_intervals = {}

    return PollingConfig(
        max_wait_seconds=int(raw.get("max_wait_seconds", 150)),
        default_interval=float(raw.get("default_interval", 3.0)),
        speech_interval=float(service_intervals.get("speech", 2.0)),
        music_interval=float(service_intervals.get("music", 5.0)),
        video_interval=float(service_intervals.get("video", 3.0)),
    )
