from __future__ import annotations

import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Quasar.cai import ArtifactInput, HealthStatus


def test_package_has_minimal_core_and_real_optional_dependency_groups():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["dependencies"] == []
    extras = project["optional-dependencies"]
    assert set(extras) >= {
        "langchain", "redis", "mongo", "sql", "object-recognition",
        "media", "providers", "all", "dev",
    }
    assert any(item.startswith("redis") for item in extras["redis"])
    assert any(item.startswith("pymongo") for item in extras["mongo"])
    assert any(item.startswith("sqlalchemy") for item in extras["sql"])
    assert any(item.startswith("sqlite-vec") for item in extras["object-recognition"])
    assert set(extras["all"]) >= set().union(*(
        set(value) for name, value in extras.items() if name not in {"all", "dev"}
    ))


def test_development_requirements_install_declared_extras_instead_of_raw_dependencies():
    root = Path(__file__).resolve().parents[1]
    requirements = (root / "requirements.txt").read_text(encoding="utf-8")

    assert requirements.splitlines() == ["-e .[all,dev]"]


def test_local_artifact_adapter_only_touches_disk_after_explicit_start(tmp_path):
    from Quasar.adapters.artifacts import LocalFileArtifactStore

    root = tmp_path / "artifacts"
    store = LocalFileArtifactStore(root)
    assert not root.exists()

    store.start()
    reference = store.put(ArtifactInput(b"result", "result.txt", "text/plain"))

    assert root.is_dir()
    assert store.open(reference).read() == b"result"
    assert store.health().status is HealthStatus.HEALTHY
    assert store.delete(reference) is True
    store.close()


def test_sqlite_vector_adapter_is_lazy_and_does_not_import_optional_driver(tmp_path):
    from Quasar.adapters.vector import SQLiteVectorStore

    database = tmp_path / "vectors.db"
    before_modules = set(sys.modules)
    store = SQLiteVectorStore(database, vector_dim=3)

    assert not database.exists()
    assert "sqlite_vec" not in set(sys.modules) - before_modules
    assert store.health().status is HealthStatus.DEGRADED


def test_sqlite_vector_adapter_round_trip(tmp_path):
    from Quasar.adapters.vector import SQLiteVectorStore
    from Quasar.cai import VectorQuery, VectorRecord

    store = SQLiteVectorStore(tmp_path / "vectors.db", vector_dim=3)
    store.start()
    assert store.health().status is HealthStatus.HEALTHY
    store.upsert([
        VectorRecord("alpha", (1.0, 0.0, 0.0), {"label": "Alpha"}),
        VectorRecord("beta", (0.0, 1.0, 0.0), {"label": "Beta"}),
    ])

    matches = store.search(VectorQuery((1.0, 0.0, 0.0), limit=2))

    assert [match.key for match in matches] == ["alpha", "beta"]
    assert matches[0].metadata == {"label": "Alpha"}
    assert matches[0].score > matches[1].score
    assert store.delete(["alpha"]) == 1
    assert [match.key for match in store.search(VectorQuery((1.0, 0.0, 0.0)))] == ["beta"]
    store.close()
    assert store.health().status is HealthStatus.DEGRADED


def test_object_recognition_is_disabled_by_default():
    from Quasar.ai_modules.object_recognition.configs.dataclasses import RecognitionConfig

    assert RecognitionConfig().enable is False
