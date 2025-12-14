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
    DummyChromaClient,
    DummyChromaVectorStore,
    DummyEmbedManager,
    DummyIndexSchema,
    DummyMultiModalVectorStoreIndex,
    DummyPGVectorStore,
    DummyRedisVectorStore,
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
    vector_cfg = VectorStoreConfig(chroma_persist_dir=persist_dir)
    vector_cfg.pgvector_password = "secret"
    stub = SimpleNamespace(general=general, vector_store=vector_cfg)
    return cast(ConfigManager, stub)


def _make_docstore() -> DocumentStoreManager:
    return cast(DocumentStoreManager, SimpleNamespace())


@pytest.fixture(autouse=True)
def patch_vector_backends(monkeypatch):
    monkeypatch.setattr(
        "llama_index.vector_stores.postgres.PGVectorStore", DummyPGVectorStore
    )
    monkeypatch.setattr("chromadb.HttpClient", DummyChromaClient)
    monkeypatch.setattr("chromadb.PersistentClient", DummyChromaClient)
    monkeypatch.setattr(
        "llama_index.vector_stores.chroma.ChromaVectorStore", DummyChromaVectorStore
    )
    monkeypatch.setattr(
        "llama_index.vector_stores.redis.RedisVectorStore", DummyRedisVectorStore
    )
    monkeypatch.setattr("redisvl.schema.IndexSchema", DummyIndexSchema)
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


def test_create_vector_store_manager_chroma_http(tmp_path):
    cfg = _make_cfg(VectorStoreProvider.CHROMA, tmp_path)
    cfg.vector_store.chroma_host = "localhost"
    cfg.vector_store.chroma_port = 8000
    embed = DummyEmbedManager({Modality.TEXT: 64})

    manager = create_vector_store_manager(cfg, embed, _make_docstore())
    cont = manager.get_container(Modality.TEXT)
    assert cont.provider_name == VectorStoreProvider.CHROMA
    store = cast(DummyChromaVectorStore, cont.store)
    assert store.collection.name == cont.table_name


def test_create_vector_store_manager_redis(tmp_path):
    cfg = _make_cfg(VectorStoreProvider.REDIS, tmp_path)
    embed = DummyEmbedManager({Modality.TEXT: 32})

    manager = create_vector_store_manager(cfg, embed, _make_docstore())
    cont = manager.get_container(Modality.TEXT)
    assert cont.provider_name == VectorStoreProvider.REDIS
    store = cast(DummyRedisVectorStore, cont.store)
    assert "redis://" in store.redis_url


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
