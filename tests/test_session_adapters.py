from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Quasar.cai import HealthStatus, SessionChange, SessionSnapshot


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.ping_count = 0
        self.closed = False

    def ping(self):
        self.ping_count += 1

    def set(self, key, value, nx=False):
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    def get(self, key):
        return self.values.get(key)

    def delete(self, key):
        return int(self.values.pop(key, None) is not None)

    def close(self):
        self.closed = True


class FakeMongoCollection:
    def __init__(self):
        self.values = {}

    def insert_one(self, document):
        key = document["session_key"]
        if key in self.values:
            raise ValueError("duplicate")
        self.values[key] = dict(document)

    def find_one(self, query):
        return self.values.get(query["session_key"])

    def replace_one(self, query, document):
        self.values[query["session_key"]] = dict(document)

    def delete_one(self, query):
        deleted = int(self.values.pop(query["session_key"], None) is not None)
        return type("Result", (), {"deleted_count": deleted})()


class FakeMongo:
    def __init__(self):
        self.collection = FakeMongoCollection()
        self.admin = type("Admin", (), {"command": lambda _self, _name: True})()
        self.closed = False

    def __getitem__(self, _database):
        collection = self.collection
        return type("Database", (), {"__getitem__": lambda _self, _name: collection})()

    def close(self):
        self.closed = True


@pytest.mark.parametrize("module_name", ["redis", "pymongo"])
def test_session_adapter_construction_does_not_import_driver_or_connect(tmp_path, module_name):
    before = set(sys.modules)
    if module_name == "redis":
        from Quasar.adapters.sessions import RedisSessionStore, RedisSessionStoreConfig
        store = RedisSessionStore(RedisSessionStoreConfig("redis://example/0"))
    else:
        from Quasar.adapters.sessions import MongoSessionStore, MongoSessionStoreConfig
        store = MongoSessionStore(MongoSessionStoreConfig("mongodb://example", "quasar"))

    assert module_name not in set(sys.modules) - before
    assert store.health().status is HealthStatus.DEGRADED


@pytest.mark.parametrize("kind", ["redis", "mongo"])
def test_session_adapters_implement_the_session_store_contract(kind):
    if kind == "redis":
        from Quasar.adapters.sessions import RedisSessionStore, RedisSessionStoreConfig
        client = FakeRedis()
        store = RedisSessionStore(RedisSessionStoreConfig("redis://example/0"), client=client)
    else:
        from Quasar.adapters.sessions import MongoSessionStore, MongoSessionStoreConfig
        client = FakeMongo()
        store = MongoSessionStore(
            MongoSessionStoreConfig("mongodb://example", "quasar"), client=client
        )

    store.start()
    store.create(SessionSnapshot("s1", "created", {"count": 1}))
    store.update(SessionChange("s1", "running", {"next": True}))

    assert store.get("s1") == SessionSnapshot(
        "s1", "running", {"count": 1, "next": True}
    )
    assert store.health().status is HealthStatus.HEALTHY
    assert store.delete("s1") is True
    assert store.get("s1") is None
    store.close()
    assert client.closed is True


def test_session_adapter_configs_reject_empty_connection_details():
    from Quasar.adapters.sessions import MongoSessionStoreConfig, RedisSessionStoreConfig

    with pytest.raises(ValueError):
        RedisSessionStoreConfig("")
    with pytest.raises(ValueError):
        MongoSessionStoreConfig("mongodb://example", "")
