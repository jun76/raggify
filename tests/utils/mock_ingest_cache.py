from __future__ import annotations

from typing import Any, ClassVar, Optional


class DummyRedisKVStore:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    @classmethod
    def from_host_and_port(cls, host: str, port: int) -> DummyRedisKVStore:
        return cls(host=host, port=port)

    def delete(self, key: str, collection: str) -> None:
        pass


class DummyPostgresKVStore:
    def __init__(self, params: dict[str, Any]) -> None:
        self.params = params

    @classmethod
    def from_params(cls, **kwargs) -> DummyPostgresKVStore:
        return cls(kwargs)

    def delete(self, key: str, collection: str) -> None:
        pass


class DummyIngestionCache:
    default_name: ClassVar[str] = "cache.json"
    last_persist_path: ClassVar[Optional[str]] = None
    last_loaded_path: ClassVar[Optional[str]] = None

    def __init__(self, cache: Any | None = None, collection: str = "") -> None:
        self.cache = cache or DummyRedisKVStore("localhost", 0)
        self.collection = collection

    @classmethod
    def from_persist_path(cls, path: str) -> DummyIngestionCache:
        inst = cls()
        cls.last_loaded_path = path
        return inst

    def persist(self, path: str) -> None:
        DummyIngestionCache.last_persist_path = path

    def clear(self) -> None:
        pass


class FakeKVStore(DummyRedisKVStore):
    def __init__(self) -> None:
        super().__init__("localhost", 6379)
        self.deleted_keys: list[tuple[str, str]] = []

    def delete(self, key: str, collection: str) -> None:
        self.deleted_keys.append((key, collection))


class FakeCache(DummyIngestionCache):
    def __init__(self) -> None:
        super().__init__(cache=FakeKVStore(), collection="collection")
        self.persist_calls: list[str] = []
        self.cleared = False

    def persist(self, path: str) -> None:
        super().persist(path)
        self.persist_calls.append(path)

    def clear(self) -> None:
        self.cleared = True
