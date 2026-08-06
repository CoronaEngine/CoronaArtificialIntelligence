from __future__ import annotations

import builtins
from pathlib import Path


def _configure_project_root(monkeypatch, project_root) -> None:
    monkeypatch.setenv("CAI_PROJECT_ROOT", str(project_root))
    for name in (
        "CAI_MEDIA_DIR",
        "CAI_MODELS_DIR",
        "CAI_SCREENSHOTS_DIR",
        "CAI_RECOGNITION_DB",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_path_resolution_does_not_create_optional_directories(
    tmp_path,
    monkeypatch,
) -> None:
    from Quasar.ai_config import paths_config

    _configure_project_root(monkeypatch, tmp_path)
    paths_config.set_paths_resolver(None)

    paths = paths_config.get_default_paths()

    assert paths.media_local_storage == tmp_path / "media"
    assert paths.assets_model_dir == tmp_path / "models"
    assert paths.screenshots_dir == tmp_path / "screenshots"
    assert not (tmp_path / "media").exists()
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "screenshots").exists()


def test_local_media_write_creates_directory_on_first_use(
    tmp_path,
    monkeypatch,
) -> None:
    from Quasar.ai_config import paths_config
    from Quasar.ai_media_resource.adapter_local import LocalStorageAdapter

    _configure_project_root(monkeypatch, tmp_path)
    paths_config.set_paths_resolver(None)
    media_dir = tmp_path / "media"
    assert not media_dir.exists()

    result = LocalStorageAdapter().save_from_base64(
        "data:application/octet-stream;base64,aGVsbG8=",
        session_id="lazy-test",
        resource_type="application/octet-stream",
    )

    saved_path = Path(result.url)
    assert media_dir.is_dir()
    assert saved_path.read_bytes() == b"hello"


def test_local_media_fallback_path_resolution_is_lazy(monkeypatch) -> None:
    from Quasar.ai_media_resource import adapter_local

    fallback = Path(adapter_local.__file__).resolve().parents[1] / "local_storage"
    assert not fallback.exists()
    original_import = builtins.__import__

    def fail_path_config_import(name, *args, **kwargs):
        if name.endswith("ai_config.paths_config"):
            raise ImportError("simulate unavailable path configuration")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_path_config_import)

    resolved = adapter_local.LocalStorageAdapter().save_path

    assert resolved == fallback
    assert not fallback.exists()


def test_environment_path_override_is_not_eagerly_created(
    tmp_path,
    monkeypatch,
) -> None:
    from Quasar.ai_config import paths_config

    custom_media = tmp_path / "custom" / "media"
    monkeypatch.setenv("CAI_MEDIA_DIR", str(custom_media))
    paths_config.set_paths_resolver(None)

    resolved = paths_config.get_project_media_dir()

    assert resolved == custom_media
    assert not custom_media.exists()
