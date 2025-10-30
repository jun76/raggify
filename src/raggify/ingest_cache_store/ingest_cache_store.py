from typing import TYPE_CHECKING

from ..config.config_manager import ConfigManager
from ..config.default_settings import IngestCacheStoreProvider
from ..config.ingest_cache_store_config import IngestCacheStoreConfig
from ..llama.core.schema import Modality
from ..logger import logger

if TYPE_CHECKING:
    from ..embed.embed_manager import EmbedManager
    from .ingest_cache_store_manager import (
        IngestCacheStoreContainer,
        IngestCacheStoreManager,
    )

__all__ = ["create_ingest_cache_store_manager"]


def create_ingest_cache_store_manager(
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
    from .ingest_cache_store_manager import IngestCacheStoreManager

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
    match cfg.general.ingest_cache_store_provider:
        case IngestCacheStoreProvider.REDIS:
            cont = _redis(cfg=cfg.ingest_cache_store, table_name=table_name)
        case IngestCacheStoreProvider.PGVECTOR:
            cont = _pgvector(cfg=cfg.ingest_cache_store, table_name=table_name)
        case _:
            logger.warning(
                f"unsupported ingest cache store: {cfg.general.ingest_cache_store_provider}, use default"
            )
            return _default(table_name)

    return cont


def _generate_table_name(cfg: ConfigManager, space_key: str) -> str:
    """テーブル名を生成する。

    Args:
        cfg (ConfigManager): 設定管理
        space_key (str): 空間キー

    Returns:
        str: テーブル名
    """
    import hashlib

    return hashlib.md5(
        f"{cfg.project_name}:{cfg.general.knowledgebase_name}:{space_key}:ics".encode()
    ).hexdigest()


# 以下、プロバイダ毎のコンテナ生成ヘルパー
def _redis(cfg: IngestCacheStoreConfig, table_name: str) -> IngestCacheStoreContainer:
    from llama_index.core.ingestion import IngestionCache
    from llama_index.storage.kvstore.redis import RedisKVStore

    from .ingest_cache_store_manager import IngestCacheStoreContainer

    return IngestCacheStoreContainer(
        provider_name=IngestCacheStoreProvider.REDIS,
        store=IngestionCache(
            cache=RedisKVStore.from_host_and_port(
                host=cfg.pgvector_host,
                port=cfg.pgvector_port,
            ),
            collection=table_name,
        ),
        table_name=table_name,
    )


def _pgvector(
    cfg: IngestCacheStoreConfig, table_name: str
) -> IngestCacheStoreContainer:
    from llama_index.core.ingestion import IngestionCache
    from llama_index.storage.kvstore.postgres import PostgresKVStore

    from .ingest_cache_store_manager import IngestCacheStoreContainer

    sec = cfg.pgvector_password
    if sec is None:
        raise ValueError("pgvector_password must be specified")

    return IngestCacheStoreContainer(
        provider_name=IngestCacheStoreProvider.PGVECTOR,
        store=IngestionCache(
            cache=PostgresKVStore.from_params(
                host=cfg.pgvector_host,
                port=str(cfg.pgvector_port),
                database=cfg.pgvector_database,
                user=cfg.pgvector_user,
                password=sec,
            ),
            collection=table_name,
        ),
        table_name=table_name,
    )


def _default(table_name: str) -> IngestCacheStoreContainer:
    from llama_index.core.ingestion import IngestionCache
    from llama_index.core.storage.kvstore import SimpleKVStore

    from .ingest_cache_store_manager import IngestCacheStoreContainer

    return IngestCacheStoreContainer(
        provider_name=IngestCacheStoreProvider.DEFAULT,
        store=IngestionCache(
            cache=SimpleKVStore(),
            collection=table_name,
        ),
        table_name=table_name,
    )
