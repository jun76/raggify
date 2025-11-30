from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from llama_index.core.ingestion import IngestionCache
from llama_index.core.schema import TextNode, TransformComponent

from raggify.config.config_manager import ConfigManager
from raggify.config.embed_config import EmbedProvider
from raggify.config.general_config import GeneralConfig
from raggify.config.ingest_cache_config import IngestCacheConfig, IngestCacheProvider
from raggify.config.ingest_config import IngestConfig
from raggify.embed.embed_manager import EmbedManager
from raggify.ingest_cache.ingest_cache import create_ingest_cache_manager
from raggify.ingest_cache.ingest_cache_manager import IngestCacheManager
from raggify.llama_like.core.schema import Modality
from tests.utils.mock_ingest_cache import (
    DummyIngestionCache,
    DummyPostgresKVStore,
    DummyRedisKVStore,
    FakeCache,
    FakeKVStore,
)


class DummyTransform(TransformComponent):
    label: str

    def __init__(self, label: str) -> None:
        object.__setattr__(self, "label", label)

    def __call__(self, nodes, **kwargs):
        return nodes


class DummyEmbedManager:
    space_key_text = "text_space"


def _make_cfg(provider: IngestCacheProvider, persist_dir) -> ConfigManager:
    general = GeneralConfig(
        knowledgebase_name="MyKB",
        ingest_cache_provider=provider,
        text_embed_provider=EmbedProvider.OPENAI,
        image_embed_provider=None,
        audio_embed_provider=None,
        video_embed_provider=None,
    )
    ingest_cache = IngestCacheConfig()
    ingest = IngestConfig(pipe_persist_dir=persist_dir)
    stub = SimpleNamespace(
        general=general,
        ingest_cache=ingest_cache,
        ingest=ingest,
    )
    return cast(ConfigManager, stub)


@pytest.fixture(autouse=True)
def patch_ingest_cache(monkeypatch):
    monkeypatch.setattr(
        "llama_index.storage.kvstore.redis.RedisKVStore", DummyRedisKVStore
    )
    monkeypatch.setattr(
        "llama_index.storage.kvstore.postgres.PostgresKVStore", DummyPostgresKVStore
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
    cfg = _make_cfg(IngestCacheProvider.LOCAL, tmp_path)
    return create_ingest_cache_manager(cfg, cast(EmbedManager, DummyEmbedManager()))


def test_name_and_modality(tmp_path):
    manager = _build_manager(tmp_path)
    assert manager.name == "local"
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
    transforms = [DummyTransform("t1")]

    def fake_hash(nodes, transform):
        return f"{transform.label}:{len(nodes)}"

    monkeypatch.setattr(
        "llama_index.core.ingestion.pipeline.get_transformation_hash", fake_hash
    )

    manager.delete(Modality.TEXT, nodes, transforms, persist_dir=tmp_path)

    assert container.cache != None
    kv = cast(FakeKVStore, cache.cache)
    assert kv.deleted_keys == [("t1:1", cache.collection)]
    assert cache.persist_calls


def test_delete_handles_no_cache(tmp_path):
    manager = _build_manager(tmp_path)
    container = manager.get_container(Modality.TEXT)
    container.cache = None
    manager.delete(Modality.TEXT, [], [], persist_dir=tmp_path)


def test_delete_all_clears_each_cache(tmp_path):
    manager = _build_manager(tmp_path)
    container = manager.get_container(Modality.TEXT)
    container.cache = cast(IngestionCache, FakeCache())
    cache = cast(FakeCache, container.cache)

    persist_dir = Path(tmp_path)
    manager.delete_all(persist_dir=persist_dir)

    assert cache is not None
    assert cache.cleared is True
    assert cache.persist_calls == [str(persist_dir / DummyIngestionCache.default_name)]
