from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from raggify.config.config_manager import ConfigManager
from raggify.config.embed_config import EmbedProvider
from raggify.config.general_config import GeneralConfig
from raggify.config.ingest_cache_config import IngestCacheConfig, IngestCacheProvider
from raggify.embed.embed_manager import EmbedManager
from raggify.ingest_cache.ingest_cache import create_ingest_cache_manager
from raggify.llama_like.core.schema import Modality
from tests.utils.mock_ingest_cache import (
    DummyIngestionCache,
    DummyPostgresKVStore,
)

from .config import configure_test_env

configure_test_env()


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

    DummyIngestionCache.last_persist_path = None
    DummyIngestionCache.last_loaded_path = None
    monkeypatch.setattr(
        "llama_index.storage.kvstore.postgres.PostgresKVStore",
        DummyPostgresKVStore,
        raising=False,
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


def _make_cfg(provider: IngestCacheProvider) -> ConfigManager:
    general = GeneralConfig(
        knowledgebase_name="My KB",
        ingest_cache_provider=provider,
        text_embed_provider=None,
        image_embed_provider=None,
        audio_embed_provider=None,
        video_embed_provider=None,
    )
    ingest_cache = IngestCacheConfig()
    stub = SimpleNamespace(general=general, ingest_cache=ingest_cache)
    return cast(ConfigManager, stub)


def test_create_ingest_cache_manager_postgres(tmp_path):
    cfg = _make_cfg(IngestCacheProvider.POSTGRES)
    cfg.general.image_embed_provider = EmbedProvider.COHERE

    manager = create_ingest_cache_manager(cfg, cast(EmbedManager, DummyEmbedManager()))

    cont = manager.get_container(Modality.IMAGE)
    assert cont.provider_name == IngestCacheProvider.POSTGRES
    assert cont.cache != None
    assert isinstance(cont.cache.cache, DummyPostgresKVStore)
    assert cont.cache.collection.endswith("_ic")

def test_create_ingest_cache_manager_requires_provider(tmp_path):
    cfg = _make_cfg(IngestCacheProvider.POSTGRES)
    manager_embed = cast(EmbedManager, DummyEmbedManager())
    with pytest.raises(RuntimeError):
        create_ingest_cache_manager(cfg, manager_embed)
