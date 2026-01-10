from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.config_manager import ConfigManager
from ..config.document_store_config import DocumentStoreConfig, DocumentStoreProvider
from ..core.const import EXTRA_PKG_NOT_FOUND_MSG, PJNAME_ALIAS
from ..core.utils import sanitize_str

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
        case DocumentStoreProvider.POSTGRES:
            return _postgres(cfg=cfg.document_store, table_name=table_name)
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
