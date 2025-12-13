from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..config.config_manager import ConfigManager
from ..config.ingest_cache_config import IngestCacheConfig, IngestCacheProvider
from ..core.const import EXTRA_PKG_NOT_FOUND_MSG, PJNAME_ALIAS
from ..core.utils import sanitize_str
from ..llama_like.core.schema import Modality
from ..logger import logger

if TYPE_CHECKING:
    from ..embed.embed_manager import EmbedManager
    from .ingest_cache_manager import IngestCacheContainer, IngestCacheManager

__all__ = ["create_ingest_cache_manager"]


def create_ingest_cache_manager(
    cfg: ConfigManager, embed: EmbedManager
) -> IngestCacheManager:
    """Create an ingest cache manager instance.

    Args:
        cfg (ConfigManager): Config manager.
        embed (EmbedManager): Embedding manager.

    Raises:
        RuntimeError: If instantiation fails or providers are missing.

    Returns:
        IngestCacheManager: Ingest cache manager.
    """
    from .ingest_cache_manager import IngestCacheManager

    try:
        conts: dict[Modality, IngestCacheContainer] = {}
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

        if cfg.general.video_embed_provider:
            conts[Modality.VIDEO] = _create_container(
                cfg=cfg, space_key=embed.space_key_video
            )
    except Exception as e:
        raise RuntimeError(f"failed to create ingest cache: {e}") from e

    if not conts:
        raise RuntimeError("no embedding providers are specified")

    return IngestCacheManager(conts)


def _create_container(cfg: ConfigManager, space_key: str) -> IngestCacheContainer:
    """Create a container for each space key.

    Args:
        cfg (ConfigManager): Config manager.
        space_key (str): Space key.

    Raises:
        RuntimeError: When the provider is unsupported.

    Returns:
        IngestCacheContainer: Container instance.
    """
    table_name = _generate_table_name(cfg, space_key)
    match cfg.general.ingest_cache_provider:
        case IngestCacheProvider.REDIS:
            return _redis(cfg=cfg.ingest_cache, table_name=table_name)
        case IngestCacheProvider.POSTGRES:
            return _postgres(cfg=cfg.ingest_cache, table_name=table_name)
        case IngestCacheProvider.LOCAL:
            return _local(persist_dir=cfg.pipeline.persist_dir, table_name=table_name)
        case _:
            raise RuntimeError(
                f"unsupported ingest cache: {cfg.general.ingest_cache_provider}"
            )


def _generate_table_name(cfg: ConfigManager, space_key: str) -> str:
    """Generate a table name.

    Args:
        cfg (ConfigManager): Config manager.
        space_key (str): Space key.

    Raises:
        ValueError: When the table name is too long.

    Returns:
        str: Table name.
    """
    return sanitize_str(
        f"{PJNAME_ALIAS}_{cfg.general.knowledgebase_name}_{space_key}_ic"
    )


# Container factory helpers per provider
def _redis(cfg: IngestCacheConfig, table_name: str) -> IngestCacheContainer:
    from llama_index.core.ingestion import IngestionCache

    try:
        from llama_index.storage.kvstore.redis import RedisKVStore  # type: ignore
    except ImportError:
        raise ImportError(
            EXTRA_PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-storage-docstore-redis",
                extra="redis",
                feature="RedisKVStore",
            )
        )

    from .ingest_cache_manager import IngestCacheContainer

    return IngestCacheContainer(
        provider_name=IngestCacheProvider.REDIS,
        cache=IngestionCache(
            cache=RedisKVStore.from_host_and_port(
                host=cfg.redis_host,
                port=cfg.redis_port,
            ),
            collection=table_name,
        ),
        table_name=table_name,
    )


def _postgres(cfg: IngestCacheConfig, table_name: str) -> IngestCacheContainer:
    from llama_index.core.ingestion import IngestionCache

    try:
        from llama_index.storage.kvstore.postgres import PostgresKVStore  # type: ignore
    except ImportError:
        raise ImportError(
            EXTRA_PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-storage-kvstore-postgres",
                extra="postgres",
                feature="PostgresKVStore",
            )
        )

    from .ingest_cache_manager import IngestCacheContainer

    return IngestCacheContainer(
        provider_name=IngestCacheProvider.POSTGRES,
        cache=IngestionCache(
            cache=PostgresKVStore.from_params(
                host=cfg.postgres_host,
                port=str(cfg.postgres_port),
                database=cfg.postgres_database,
                user=cfg.postgres_user,
                password=cfg.postgres_password,
                table_name=table_name,
            ),
            collection=table_name,
        ),
        table_name=table_name,
    )


def _local(persist_dir: Path, table_name: str) -> IngestCacheContainer:
    from llama_index.core.ingestion.cache import DEFAULT_CACHE_NAME, IngestionCache

    from .ingest_cache_manager import IngestCacheContainer

    cache = IngestionCache()
    if persist_dir.exists():
        try:
            cache = IngestionCache.from_persist_path(
                str(persist_dir / DEFAULT_CACHE_NAME)
            )
            logger.info(f"loaded from persist dir: {persist_dir}")
        except Exception as e:
            logger.warning(f"failed to load persist dir: {e}")
            cache = IngestionCache()

    return IngestCacheContainer(
        provider_name=IngestCacheProvider.LOCAL,
        cache=cache,
        table_name=table_name,
    )
