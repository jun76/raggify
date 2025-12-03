from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from raggify.config.config_manager import ConfigManager
from raggify.config.document_store_config import (
    DocumentStoreConfig,
    DocumentStoreProvider,
)
from raggify.config.general_config import GeneralConfig
from raggify.config.ingest_config import IngestConfig
from raggify.document_store.document_store import create_document_store_manager
from tests.utils.mock_document_store import (
    DummyPostgresDocumentStore,
    DummyRedisDocumentStore,
    DummySimpleDocumentStore,
)

from .config import configure_test_env

configure_test_env()


@pytest.fixture(autouse=True)
def patch_document_store_classes(monkeypatch):
    monkeypatch.setattr(
        "llama_index.storage.docstore.redis.RedisDocumentStore", DummyRedisDocumentStore
    )
    monkeypatch.setattr(
        "llama_index.storage.docstore.postgres.PostgresDocumentStore",
        DummyPostgresDocumentStore,
    )
    monkeypatch.setattr(
        "llama_index.core.storage.docstore.SimpleDocumentStore",
        DummySimpleDocumentStore,
    )


def _make_cfg(provider: DocumentStoreProvider, persist_dir) -> ConfigManager:
    general = GeneralConfig(
        document_store_provider=provider, knowledgebase_name="My KB"
    )
    document_store = DocumentStoreConfig()
    ingest = IngestConfig(pipe_persist_dir=persist_dir)
    stub = SimpleNamespace(
        general=general, document_store=document_store, ingest=ingest
    )
    return cast(ConfigManager, stub)


def test_create_document_store_manager_redis(tmp_path):
    cfg = _make_cfg(DocumentStoreProvider.REDIS, tmp_path)
    manager = create_document_store_manager(cfg)

    assert manager.name == DocumentStoreProvider.REDIS
    assert isinstance(manager.store, DummyRedisDocumentStore)
    assert manager.store.namespace.endswith("_doc")


def test_create_document_store_manager_postgres(tmp_path):
    cfg = _make_cfg(DocumentStoreProvider.POSTGRES, tmp_path)
    manager = create_document_store_manager(cfg)

    assert manager.name == DocumentStoreProvider.POSTGRES
    assert isinstance(manager.store, DummyPostgresDocumentStore)
    assert manager.store.params["table_name"].endswith("_doc")


def test_create_document_store_manager_local_loads_existing(tmp_path):
    persist_dir = tmp_path / "kb"
    persist_dir.mkdir()

    cfg = _make_cfg(DocumentStoreProvider.LOCAL, persist_dir)
    manager = create_document_store_manager(cfg)

    assert manager.name == DocumentStoreProvider.LOCAL
    assert isinstance(manager.store, DummySimpleDocumentStore)
    assert manager.store.loaded_from == str(persist_dir)
    assert manager.table_name is None


def test_create_document_store_manager_local_fallback_on_error(monkeypatch, tmp_path):
    persist_dir = tmp_path / "kb2"
    persist_dir.mkdir()

    def broken_from_dir(cls, path):
        raise RuntimeError("failed")

    monkeypatch.setattr(
        "llama_index.core.storage.docstore.SimpleDocumentStore.from_persist_dir",
        broken_from_dir,
    )
    cfg = _make_cfg(DocumentStoreProvider.LOCAL, persist_dir)
    manager = create_document_store_manager(cfg)

    assert manager.name == DocumentStoreProvider.LOCAL
    assert isinstance(manager.store, DummySimpleDocumentStore)
    assert manager.store.loaded_from is None
