from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..config.config_manager import ConfigManager
from ..config.document_store_config import DocumentStoreConfig, DocumentStoreProvider
from ..core.const import PROJECT_NAME
from ..core.util import sanitize_str
from ..logger import logger

if TYPE_CHECKING:
    from .document_store_manager import DocumentStoreManager

__all__ = ["create_document_store_manager"]


def create_document_store_manager(cfg: ConfigManager) -> DocumentStoreManager:
    """同一ソースの更新用ドキュメントストア管理を生成する。

    Args:
        cfg (ConfigManager): 設定管理

    Raises:
        RuntimeError: サポート外のプロバイダ

    Returns:
        DocumentStoreManager: ドキュメントストア管理
    """
    table_name = _generate_table_name(cfg)
    match cfg.general.document_store_provider:
        case DocumentStoreProvider.REDIS:
            return _redis(cfg=cfg.document_store, table_name=table_name)
        case DocumentStoreProvider.LOCAL:
            return _local(cfg.ingest.pipe_persist_dir)
        case _:
            raise RuntimeError(
                f"unsupported document store: {cfg.general.document_store_provider}"
            )


def _generate_table_name(cfg: ConfigManager) -> str:
    """テーブル名を生成する。

    Args:
        cfg (ConfigManager): 設定管理

    Raises:
        ValueError: 長すぎるテーブル名

    Returns:
        str: テーブル名
    """
    return sanitize_str(f"{PROJECT_NAME}_{cfg.general.knowledgebase_name}_doc")


# 以下、プロバイダ毎のコンテナ生成ヘルパー
def _redis(cfg: DocumentStoreConfig, table_name: str) -> DocumentStoreManager:
    from llama_index.storage.docstore.redis import RedisDocumentStore

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


def _local(persist_dir: Path) -> DocumentStoreManager:
    from llama_index.core.storage.docstore import SimpleDocumentStore

    from .document_store_manager import DocumentStoreManager

    if persist_dir.exists():
        try:
            # IngestionPipeline.persist/load の仕様に追従して、ナレッジベース毎に
            # サブディレクトリを切って区別するのでテーブル名はデフォルトのものを使用
            store = SimpleDocumentStore.from_persist_dir(str(persist_dir))
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
