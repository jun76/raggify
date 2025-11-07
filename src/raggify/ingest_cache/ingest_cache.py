from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..config.config_manager import ConfigManager
from ..config.ingest_cache_config import IngestCacheConfig, IngestCacheStoreProvider
from ..core.const import PROJECT_NAME
from ..core.util import sanitize_str
from ..llama.core.schema import Modality
from ..logger import logger

if TYPE_CHECKING:
    from ..embed.embed_manager import EmbedManager
    from .ingest_cache_manager import IngestCacheStoreContainer, IngestCacheStoreManager

__all__ = ["create_ingest_cache_manager"]


def create_ingest_cache_manager(
    cfg: ConfigManager, embed: EmbedManager
) -> IngestCacheStoreManager:
    """インジェストキャッシュ管理のインスタンスを生成する。

    Args:
        cfg (ConfigManager): 設定管理
        embed (EmbedManager): 埋め込み管理

    Raises:
        RuntimeError: インスタンス生成に失敗またはプロバイダ指定漏れ

    Returns:
        IngestCacheStoreManager: インジェストキャッシュ管理
    """
    from .ingest_cache_manager import IngestCacheStoreManager

    try:
        conts: dict[Modality, IngestCacheStoreContainer] = {}
        if cfg.general.text_embed_provider:
            conts[Modality.TEXT] = _create_container(
                cfg=cfg, space_key=embed.space_key_text
            )

        if cfg.general.image_embed_provider:
            conts[Modality.IMAGE] = _create_container(
                cfg=cfg, space_key=embed.space_key_image
            )

        if cfg.general.audio_embed_provider:
            conts[Modality.AUDIO] = _create_container(
                cfg=cfg, space_key=embed.space_key_audio
            )
    except Exception as e:
        raise RuntimeError(f"failed to create vector store: {e}") from e

    if not conts:
        raise RuntimeError("no embedding providers are specified")

    return IngestCacheStoreManager(conts)


def _create_container(cfg: ConfigManager, space_key: str) -> IngestCacheStoreContainer:
    """空間キー毎のコンテナを生成する。

    Args:
        cfg (ConfigManager): 設定管理
        space_key (str): 空間キー

    Raises:
        RuntimeError: サポート外のプロバイダ

    Returns:
        IngestCacheStoreContainer: コンテナ
    """
    table_name = _generate_table_name(cfg, space_key)
    match cfg.general.ingest_cache_provider:
        case IngestCacheStoreProvider.REDIS:
            return _redis(cfg=cfg.ingest_cache, table_name=table_name)
        case IngestCacheStoreProvider.LOCAL:
            return _local(
                persist_dir=cfg.ingest.pipe_persist_dir, table_name=table_name
            )
        case _:
            raise RuntimeError(
                f"unsupported ingest cache: {cfg.general.ingest_cache_provider}"
            )


def _generate_table_name(cfg: ConfigManager, space_key: str) -> str:
    """テーブル名を生成する。

    Args:
        cfg (ConfigManager): 設定管理
        space_key (str): 空間キー

    Raises:
        ValueError: 長すぎるテーブル名

    Returns:
        str: テーブル名
    """
    return sanitize_str(
        f"{PROJECT_NAME}_{cfg.general.knowledgebase_name}_{space_key}_ic"
    )


# 以下、プロバイダ毎のコンテナ生成ヘルパー
def _redis(cfg: IngestCacheConfig, table_name: str) -> IngestCacheStoreContainer:
    from llama_index.core.ingestion import IngestionCache
    from llama_index.storage.kvstore.redis import RedisKVStore

    from .ingest_cache_manager import IngestCacheStoreContainer

    return IngestCacheStoreContainer(
        provider_name=IngestCacheStoreProvider.REDIS,
        cache=IngestionCache(
            cache=RedisKVStore.from_host_and_port(
                host=cfg.redis_host,
                port=cfg.redis_port,
            ),
            collection=table_name,
        ),
        table_name=table_name,
    )


def _local(persist_dir: Path, table_name: str) -> IngestCacheStoreContainer:
    from llama_index.core.ingestion.cache import DEFAULT_CACHE_NAME, IngestionCache

    from .ingest_cache_manager import IngestCacheStoreContainer

    if persist_dir and persist_dir.exists():
        try:
            cache = IngestionCache.from_persist_path(
                str(persist_dir / DEFAULT_CACHE_NAME)
            )
        except Exception as e:
            logger.warning(f"failed to load persist dir: {e}")
            cache = IngestionCache()

    return IngestCacheStoreContainer(
        provider_name=IngestCacheStoreProvider.LOCAL,
        cache=cache,
        table_name=table_name,
    )
