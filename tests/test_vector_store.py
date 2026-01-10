from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from raggify.config.config_manager import ConfigManager
from raggify.config.embed_config import EmbedProvider
from raggify.config.general_config import GeneralConfig
from raggify.config.vector_store_config import VectorStoreConfig, VectorStoreProvider
from raggify.document_store.document_store_manager import DocumentStoreManager
from raggify.embed.embed_manager import Modality
from raggify.vector_store.vector_store import create_vector_store_manager
from tests.utils.mock_vector_store import (
    DummyEmbedManager,
    DummyMultiModalVectorStoreIndex,
    DummyPGVectorStore,
    DummyVectorStoreIndex,
    FakeVectorStore,
)

from .config import configure_test_env

configure_test_env()


def _make_cfg(provider: VectorStoreProvider, persist_dir) -> ConfigManager:
    general = GeneralConfig(
        knowledgebase_name="My KB",
        vector_store_provider=provider,
        text_embed_provider=EmbedProvider.OPENAI,
        image_embed_provider=None,
        audio_embed_provider=None,
        video_embed_provider=None,
    )
    vector_cfg = VectorStoreConfig()
    vector_cfg.pgvector_password = "secret"
    stub = SimpleNamespace(general=general, vector_store=vector_cfg)
    return cast(ConfigManager, stub)


def _make_docstore() -> DocumentStoreManager:
    return cast(DocumentStoreManager, SimpleNamespace(store=SimpleNamespace()))


@pytest.fixture(autouse=True)
def patch_vector_backends(monkeypatch):
    import sys
    import types

    module_name = "llama_index.vector_stores.postgres"
    dummy_module = types.ModuleType(module_name)
    dummy_module.PGVectorStore = DummyPGVectorStore
    sys.modules.setdefault(module_name, dummy_module)

    import llama_index.vector_stores as vector_stores

    monkeypatch.setattr(vector_stores, "postgres", dummy_module, raising=False)

    monkeypatch.setattr(
        "llama_index.vector_stores.postgres.PGVectorStore", DummyPGVectorStore
    )
    monkeypatch.setattr(
        "llama_index.core.indices.VectorStoreIndex", DummyVectorStoreIndex
    )
    monkeypatch.setattr(
        "llama_index.core.indices.multi_modal.MultiModalVectorStoreIndex",
        DummyMultiModalVectorStoreIndex,
    )


def test_create_vector_store_manager_pgvector(tmp_path):
    cfg = _make_cfg(VectorStoreProvider.PGVECTOR, tmp_path)
    embed = DummyEmbedManager({Modality.TEXT: 128})
    manager = create_vector_store_manager(cfg, embed, _make_docstore())

    assert manager.modality == {Modality.TEXT}
    cont = manager.get_container(Modality.TEXT)
    assert cont.provider_name == VectorStoreProvider.PGVECTOR
    assert cont.table_name.endswith("_vec")
    store = cast(DummyPGVectorStore, cont.store)
    assert store.params["table_name"] == cont.table_name


def test_vector_store_manager_delete_operations(tmp_path):
    cfg = _make_cfg(VectorStoreProvider.PGVECTOR, tmp_path)
    embed = DummyEmbedManager({Modality.TEXT: 16})
    manager = create_vector_store_manager(cfg, embed, _make_docstore())

    # Replace underlying store with FakeVectorStore to track ops
    cont = manager.get_container(Modality.TEXT)
    cont.store = cast(BasePydanticVectorStore, FakeVectorStore())
    store = cast(FakeVectorStore, cont.store)

    manager.delete_nodes({"doc1", "doc2"})
    assert store.deleted == {"doc1", "doc2"}

    assert manager.delete_all() is True
    assert store.cleared is True
