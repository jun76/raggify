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
from raggify.document_store.document_store import create_document_store_manager
from tests.utils.mock_document_store import DummyPostgresDocumentStore

from .config import configure_test_env

configure_test_env()


@pytest.fixture(autouse=True)
def patch_document_store_classes(monkeypatch):
    import sys
    import types

    module_name = "llama_index.storage.docstore.postgres"
    dummy_module = types.ModuleType(module_name)
    dummy_module.PostgresDocumentStore = DummyPostgresDocumentStore
    sys.modules.setdefault(module_name, dummy_module)

    import llama_index.storage.docstore as docstore

    monkeypatch.setattr(docstore, "postgres", dummy_module, raising=False)


def _make_cfg(provider: DocumentStoreProvider) -> ConfigManager:
    general = GeneralConfig(
        document_store_provider=provider, knowledgebase_name="My KB"
    )
    document_store = DocumentStoreConfig()
    stub = SimpleNamespace(general=general, document_store=document_store)
    return cast(ConfigManager, stub)


def test_create_document_store_manager_postgres(tmp_path):
    cfg = _make_cfg(DocumentStoreProvider.POSTGRES)
    manager = create_document_store_manager(cfg)

    assert manager.name == DocumentStoreProvider.POSTGRES
    assert isinstance(manager.store, DummyPostgresDocumentStore)
    assert manager.store.params["table_name"].endswith("_doc")
