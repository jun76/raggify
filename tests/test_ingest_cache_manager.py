from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from llama_index.core.ingestion import IngestionCache
from llama_index.core.schema import TextNode, TransformComponent

from raggify.config.config_manager import ConfigManager
from raggify.config.embed_config import EmbedProvider
from raggify.config.general_config import GeneralConfig
from raggify.config.ingest_cache_config import IngestCacheConfig, IngestCacheProvider
from raggify.embed.embed_manager import EmbedManager
from raggify.ingest_cache.ingest_cache import create_ingest_cache_manager
from raggify.ingest_cache.ingest_cache_manager import IngestCacheManager
from raggify.llama_like.core.schema import Modality
from tests.utils.mock_ingest_cache import (
    DummyIngestionCache,
    DummyPostgresKVStore,
    FakeCache,
    FakeKVStore,
)

from .config import configure_test_env

configure_test_env()


class DummyTransform(TransformComponent):
    label: str

    def __init__(self, label: str) -> None:
        object.__setattr__(self, "label", label)

    def __call__(self, nodes, **kwargs):
        return nodes


class DummyEmbedManager:
    space_key_text = "text_space"


def _make_cfg(provider: IngestCacheProvider) -> ConfigManager:
    general = GeneralConfig(
        knowledgebase_name="MyKB",
        ingest_cache_provider=provider,
        text_embed_provider=EmbedProvider.OPENAI,
        image_embed_provider=None,
        audio_embed_provider=None,
        video_embed_provider=None,
    )
    ingest_cache = IngestCacheConfig()
    stub = SimpleNamespace(general=general, ingest_cache=ingest_cache)
    return cast(ConfigManager, stub)


@pytest.fixture(autouse=True)
def patch_ingest_cache(monkeypatch):
    import sys
    import types

    module_name = "llama_index.storage.kvstore.postgres"
    dummy_module = types.ModuleType(module_name)
    dummy_module.PostgresKVStore = DummyPostgresKVStore
    sys.modules.setdefault(module_name, dummy_module)

    import llama_index.storage.kvstore as kvstore

    monkeypatch.setattr(kvstore, "postgres", dummy_module, raising=False)

    monkeypatch.setattr(
        "llama_index.storage.kvstore.postgres.PostgresKVStore",
        DummyPostgresKVStore,
        raising=False,
    )
    monkeypatch.setattr(
        "llama_index.core.ingestion.IngestionCache", DummyIngestionCache
    )
    monkeypatch.setattr(
        "llama_index.core.ingestion.cache.IngestionCache", DummyIngestionCache
    )
    monkeypatch.setattr(
        "llama_index.core.ingestion.cache.DEFAULT_CACHE_NAME",
        DummyIngestionCache.default_name,
        raising=False,
    )


def _build_manager(tmp_path) -> IngestCacheManager:
    cfg = _make_cfg(IngestCacheProvider.POSTGRES)
    return create_ingest_cache_manager(cfg, cast(EmbedManager, DummyEmbedManager()))


def test_name_and_modality(tmp_path):
    manager = _build_manager(tmp_path)
    assert manager.name == str(IngestCacheProvider.POSTGRES)
    assert Modality.TEXT in manager.modality


def test_get_container_missing(tmp_path):
    manager = _build_manager(tmp_path)
    with pytest.raises(RuntimeError):
        manager.get_container(Modality.IMAGE)


def test_delete_invokes_cache_operations(monkeypatch, tmp_path):
    manager = _build_manager(tmp_path)
    container = manager.get_container(Modality.TEXT)
    container.cache = cast(IngestionCache, FakeCache())
    cache = cast(FakeCache, container.cache)

    nodes = [TextNode(text="a", id_="1")]
    transform = DummyTransform("t1")

    def fake_hash(nodes, transform):
        return f"{transform.label}:{len(nodes)}"

    monkeypatch.setattr(
        "llama_index.core.ingestion.pipeline.get_transformation_hash", fake_hash
    )

    manager.delete_nodes(Modality.TEXT, nodes, transform)

    assert container.cache != None
    kv = cast(FakeKVStore, cache.cache)
    assert kv.deleted_keys == [("t1:1", cache.collection)]


def test_delete_handles_no_cache(tmp_path):
    manager = _build_manager(tmp_path)
    container = manager.get_container(Modality.TEXT)
    container.cache = None
    manager.delete_nodes(Modality.TEXT, [], DummyTransform("noop"))


def test_delete_all_clears_each_cache(tmp_path):
    manager = _build_manager(tmp_path)
    container = manager.get_container(Modality.TEXT)
    container.cache = cast(IngestionCache, FakeCache())
    cache = cast(FakeCache, container.cache)

    manager.delete_all()

    assert cache is not None
    assert cache.cleared is True
