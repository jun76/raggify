from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from raggify.config.config_manager import ConfigManager
from raggify.config.embed_config import EmbedProvider
from raggify.config.general_config import GeneralConfig
from raggify.config.ingest_cache_config import IngestCacheConfig, IngestCacheProvider
from raggify.config.pipeline_config import PipelineConfig
from raggify.embed.embed_manager import EmbedManager
from raggify.ingest_cache.ingest_cache import create_ingest_cache_manager
from raggify.llama_like.core.schema import Modality
from tests.utils.mock_ingest_cache import (
    DummyIngestionCache,
    DummyPostgresKVStore,
    DummyRedisKVStore,
)

from .config import configure_test_env

configure_test_env()


@pytest.fixture(autouse=True)
def patch_ingest_cache(monkeypatch):
    DummyIngestionCache.last_persist_path = None
    DummyIngestionCache.last_loaded_path = None
    monkeypatch.setattr(
        "llama_index.storage.kvstore.redis.RedisKVStore",
        DummyRedisKVStore,
    )
    monkeypatch.setattr(
        "llama_index.storage.kvstore.postgres.PostgresKVStore",
        DummyPostgresKVStore,
    )
    monkeypatch.setattr(
        "llama_index.core.ingestion.IngestionCache",
        DummyIngestionCache,
    )
    monkeypatch.setattr(
        "llama_index.core.ingestion.cache.IngestionCache",
        DummyIngestionCache,
    )
    monkeypatch.setattr(
        "llama_index.core.ingestion.cache.DEFAULT_CACHE_NAME",
        DummyIngestionCache.default_name,
        raising=False,
    )


class DummyEmbedManager:
    space_key_text = "text_space"
    space_key_image = "image_space"
    space_key_audio = "audio_space"
    space_key_video = "video_space"


def _make_cfg(provider: IngestCacheProvider, persist_dir) -> ConfigManager:
    general = GeneralConfig(
        knowledgebase_name="My KB",
        ingest_cache_provider=provider,
        text_embed_provider=None,
        image_embed_provider=None,
        audio_embed_provider=None,
        video_embed_provider=None,
    )
    ingest_cache = IngestCacheConfig()
    pipeline = PipelineConfig(persist_dir=persist_dir)
    stub = SimpleNamespace(
        general=general,
        ingest_cache=ingest_cache,
        pipeline=pipeline,
    )
    return cast(ConfigManager, stub)


def test_create_ingest_cache_manager_redis(tmp_path):
    cfg = _make_cfg(IngestCacheProvider.REDIS, tmp_path)
    cfg.general.text_embed_provider = EmbedProvider.OPENAI

    manager = create_ingest_cache_manager(cfg, cast(EmbedManager, DummyEmbedManager()))

    assert Modality.TEXT in manager.modality
    cont = manager.get_container(Modality.TEXT)
    assert cont.provider_name == IngestCacheProvider.REDIS
    assert cont.table_name.endswith("_ic")
    assert cont.cache != None
    assert isinstance(cont.cache.cache, DummyRedisKVStore)


def test_create_ingest_cache_manager_postgres(tmp_path):
    cfg = _make_cfg(IngestCacheProvider.POSTGRES, tmp_path)
    cfg.general.image_embed_provider = EmbedProvider.COHERE

    manager = create_ingest_cache_manager(cfg, cast(EmbedManager, DummyEmbedManager()))

    cont = manager.get_container(Modality.IMAGE)
    assert cont.provider_name == IngestCacheProvider.POSTGRES
    assert cont.cache != None
    assert isinstance(cont.cache.cache, DummyPostgresKVStore)
    assert cont.cache.collection.endswith("_ic")


def test_create_ingest_cache_manager_local_load(tmp_path):
    persist_dir = tmp_path / "kb"
    persist_dir.mkdir()
    cfg = _make_cfg(IngestCacheProvider.LOCAL, persist_dir)
    cfg.general.audio_embed_provider = EmbedProvider.BEDROCK

    manager = create_ingest_cache_manager(cfg, cast(EmbedManager, DummyEmbedManager()))

    cont = manager.get_container(Modality.AUDIO)
    assert cont.provider_name == IngestCacheProvider.LOCAL
    assert DummyIngestionCache.last_loaded_path == str(
        persist_dir / DummyIngestionCache.default_name
    )


def test_create_ingest_cache_manager_local_fallback(monkeypatch, tmp_path):
    persist_dir = tmp_path / "kb"
    persist_dir.mkdir()
    cfg = _make_cfg(IngestCacheProvider.LOCAL, persist_dir)
    cfg.general.video_embed_provider = EmbedProvider.BEDROCK

    def broken_from_path(path):
        raise RuntimeError("failed")

    monkeypatch.setattr(
        "llama_index.core.ingestion.cache.IngestionCache.from_persist_path",
        broken_from_path,
    )

    manager = create_ingest_cache_manager(cfg, cast(EmbedManager, DummyEmbedManager()))

    assert DummyIngestionCache.last_loaded_path is None


def test_create_ingest_cache_manager_requires_provider(tmp_path):
    cfg = _make_cfg(IngestCacheProvider.LOCAL, tmp_path)
    manager_embed = cast(EmbedManager, DummyEmbedManager())
    with pytest.raises(RuntimeError):
        create_ingest_cache_manager(cfg, manager_embed)
