from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Quasar.cai import VectorMatch


class FakeVectorStore:
    def __init__(self):
        self.records = {}
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def upsert(self, records):
        self.records.update({record.key: record for record in records})

    def search(self, query):
        return [
            VectorMatch(key, 1.0, record.metadata)
            for key, record in list(self.records.items())[:query.limit]
        ]

    def delete(self, keys):
        for key in keys:
            self.records.pop(key, None)

    def close(self):
        self.closed = True


def test_object_recognition_facade_uses_injected_vector_store_without_sqlite(tmp_path):
    from Quasar.ai_modules.object_recognition.tools.vector_db import VectorDB

    store = FakeVectorStore()
    database = VectorDB(vector_dim=3, vector_store=store)
    assert list(tmp_path.iterdir()) == []

    database.insert_object(
        "cup", np.array([1.0, 0.0, 0.0]), name="杯子", category="餐具"
    )

    assert store.started is True
    assert database.search(np.array([1.0, 0.0, 0.0]))[0]["object_id"] == "cup"
    assert database.get_object("cup")["name"] == "杯子"
    database.close()
    assert store.closed is True
