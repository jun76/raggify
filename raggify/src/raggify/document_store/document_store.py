from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..config.config_manager import ConfigManager
from ..config.document_store_config import DocumentStoreConfig, DocumentStoreProvider
from ..core.const import EXTRA_PKG_NOT_FOUND_MSG, PJNAME_ALIAS
from ..core.utils import sanitize_str
from ..logger import logger

if TYPE_CHECKING:
    from .document_store_manager import DocumentStoreManager

__all__ = ["create_document_store_manager"]


def create_document_store_manager(cfg: ConfigManager) -> DocumentStoreManager:
    """Create the document store manager for tracking source updates.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        RuntimeError: If the provider is unsupported.

    Returns:
        DocumentStoreManager: Document store manager.
    """
    table_name = _generate_table_name(cfg)
    match cfg.general.document_store_provider:
        case DocumentStoreProvider.REDIS:
            return _redis(cfg=cfg.document_store, table_name=table_name)
        case DocumentStoreProvider.POSTGRES:
            return _postgres(cfg=cfg.document_store, table_name=table_name)
        case DocumentStoreProvider.LOCAL:
            return _local(cfg.pipeline.persist_dir)
        case _:
            raise RuntimeError(
                f"unsupported document store: {cfg.general.document_store_provider}"
            )


def _generate_table_name(cfg: ConfigManager) -> str:
    """Generate the table name.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        ValueError: If the table name is too long.

    Returns:
        str: Table name.
    """
    return sanitize_str(f"{PJNAME_ALIAS}_{cfg.general.knowledgebase_name}_doc")


# Container factory helpers per provider
def _redis(cfg: DocumentStoreConfig, table_name: str) -> DocumentStoreManager:
    try:
        from llama_index.storage.docstore.redis import (  # type: ignore
            RedisDocumentStore,
        )
    except ImportError:
        raise ImportError(
            EXTRA_PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-storage-docstore-redis",
                extra="redis",
                feature="RedisDocumentStore",
            )
        )

    from .document_store_manager import DocumentStoreManager

    return DocumentStoreManager(
        provider_name=DocumentStoreProvider.REDIS,
        store=RedisDocumentStore.from_host_and_port(
            host=cfg.redis_host,
            port=cfg.redis_port,
            namespace=table_name,
        ),
        table_name=table_name,
    )


def _postgres(cfg: DocumentStoreConfig, table_name: str) -> DocumentStoreManager:
    try:
        from llama_index.storage.docstore.postgres import (  # type: ignore
            PostgresDocumentStore,
        )
    except ImportError:
        raise ImportError(
            EXTRA_PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-storage-docstore-postgres",
                extra="postgres",
                feature="PostgresDocumentStore",
            )
        )

    from .document_store_manager import DocumentStoreManager

    return DocumentStoreManager(
        provider_name=DocumentStoreProvider.POSTGRES,
        store=PostgresDocumentStore.from_params(
            host=cfg.postgres_host,
            port=str(cfg.postgres_port),
            database=cfg.postgres_database,
            user=cfg.postgres_user,
            password=cfg.postgres_password,
            table_name=table_name,
            namespace=table_name,
        ),
        table_name=table_name,
    )


def _local(persist_dir: Path) -> DocumentStoreManager:
    from llama_index.core.storage.docstore import SimpleDocumentStore

    from .document_store_manager import DocumentStoreManager

    if persist_dir.exists():
        try:
            # Follow IngestionPipeline.persist/load:
            # separate subdirectories per knowledge base, so use default table name.
            store = SimpleDocumentStore.from_persist_dir(str(persist_dir))
            logger.info(f"loaded from persist dir: {persist_dir}")
        except Exception as e:
            logger.warning(f"failed to load persist dir {persist_dir}: {e}")
            store = SimpleDocumentStore()
    else:
        store = SimpleDocumentStore()

    return DocumentStoreManager(
        provider_name=DocumentStoreProvider.LOCAL,
        store=store,
        table_name=None,
    )
