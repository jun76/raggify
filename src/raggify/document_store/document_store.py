from typing import TYPE_CHECKING

from ..config.config_manager import ConfigManager
from ..config.default_settings import DocumentStoreProvider
from ..config.document_store_config import DocumentStoreConfig
from ..logger import logger

if TYPE_CHECKING:
    from .document_store_manager import DocumentStoreManager

__all__ = ["create_document_store_manager"]


def create_document_store_manager(cfg: ConfigManager) -> DocumentStoreManager:
    """同一ソースの更新用ドキュメントストア管理を生成する。

    Args:
        cfg (ConfigManager): 設定管理

    Returns:
        DocumentStoreManager: ドキュメントストア管理
    """
    table_name = _generate_table_name(cfg)
    match cfg.general.document_store_provider:
        case DocumentStoreProvider.REDIS:
            return _redis(cfg=cfg.document_store, table_name=table_name)
        case DocumentStoreProvider.PGVECTOR:
            return _pgvector(cfg=cfg.document_store, table_name=table_name)
        case _:
            logger.warning(
                f"unsupported vector store: {cfg.general.document_store_provider}, use default"
            )
            return _default(table_name)


def _generate_table_name(cfg: ConfigManager) -> str:
    """テーブル名を生成する。

    Args:
        cfg (ConfigManager): 設定管理

    Returns:
        str: テーブル名
    """
    import hashlib

    return hashlib.md5(
        f"{cfg.project_name}:{cfg.general.knowledgebase_name}:doc".encode()
    ).hexdigest()


# 以下、プロバイダ毎のコンテナ生成ヘルパー
def _redis(cfg: DocumentStoreConfig, table_name: str) -> DocumentStoreManager:
    from llama_index.storage.docstore.redis import RedisDocumentStore

    from .document_store_manager import DocumentStoreManager

    return DocumentStoreManager(
        provider_name=DocumentStoreProvider.REDIS,
        store=RedisDocumentStore.from_host_and_port(
            host=cfg.pgvector_host,
            port=cfg.pgvector_port,
            namespace=table_name,
        ),
        table_name=table_name,
    )


# FIXME: よく見たら pgvector じゃなくて postgres だった。config 分ける
def _pgvector(cfg: DocumentStoreConfig, table_name: str) -> DocumentStoreManager:
    from llama_index.storage.docstore.postgres import PostgresDocumentStore

    from .document_store_manager import DocumentStoreManager

    sec = cfg.pgvector_password
    if sec is None:
        raise ValueError("pgvector_password must be specified")

    return DocumentStoreManager(
        provider_name=DocumentStoreProvider.PGVECTOR,
        store=PostgresDocumentStore.from_params(
            host=cfg.pgvector_host,
            port=str(cfg.pgvector_port),
            database=cfg.pgvector_database,
            user=cfg.pgvector_user,
            password=sec,
            table_name=table_name,
        ),
        table_name=table_name,
    )


def _default(table_name: str) -> DocumentStoreManager:
    from llama_index.core.storage.docstore import SimpleDocumentStore

    from .document_store_manager import DocumentStoreManager

    return DocumentStoreManager(
        provider_name=DocumentStoreProvider.DEFAULT,
        store=SimpleDocumentStore(namespace=table_name),
        table_name=table_name,
    )
