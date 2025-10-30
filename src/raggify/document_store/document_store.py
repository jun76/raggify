from typing import TYPE_CHECKING

from ..config.config_manager import ConfigManager
from ..config.default_settings import DocumentStoreProvider
from ..config.document_store_config import DocumentStoreConfig
from ..llama.core.schema import Modality

if TYPE_CHECKING:
    from ..embed.embed_manager import EmbedManager
    from .document_store_manager import DocumentStoreContainer, DocumentStoreManager

__all__ = ["create_document_store_manager"]


def create_document_store_manager(
    cfg: ConfigManager,
    embed: EmbedManager,
) -> DocumentStoreManager:
    """ドキュメントストア管理のインスタンスを生成する。

    Args:
        cfg (ConfigManager): 設定管理
        embed (EmbedManager): 埋め込み管理

    Raises:
        RuntimeError: インスタンス生成に失敗またはプロバイダ指定漏れ

    Returns:
        DocumentStoreManager: ドキュメントストア管理
    """
    from .document_store_manager import DocumentStoreManager

    try:
        conts: dict[Modality, DocumentStoreContainer] = {}
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

    return DocumentStoreManager(conts)


def _create_container(cfg: ConfigManager, space_key: str) -> DocumentStoreContainer:
    """空間キー毎のドキュメントストアコンテナを生成する。

    Args:
        cfg (ConfigManager): 設定管理
        space_key (str): 空間キー

    Raises:
        RuntimeError: サポート外のプロバイダ

    Returns:
        DocumentStoreContainer: コンテナ
    """
    table_name = _generate_table_name(cfg, space_key)
    match cfg.general.document_store_provider:
        case DocumentStoreProvider.REDIS:
            cont = _redis(cfg=cfg.document_store, table_name=table_name)
        case DocumentStoreProvider.PGVECTOR:
            cont = _pgvector(cfg=cfg.document_store, table_name=table_name)
        case _:
            raise RuntimeError(
                f"unsupported vector store: {cfg.general.document_store_provider}"
            )

    return cont


def _generate_table_name(cfg: ConfigManager, space_key: str) -> str:
    """テーブル名を生成する。

    Args:
        cfg (ConfigManager): 設定管理
        space_key (str): 空間キー

    Returns:
        str: テーブル名
    """
    return f"{cfg.project_name}__{cfg.general.knowledgebase_name}__{space_key}"


# 以下、プロバイダ毎のコンテナ生成ヘルパー
def _redis(cfg: DocumentStoreConfig, table_name: str) -> DocumentStoreContainer:
    from llama_index.storage.docstore.redis import RedisDocumentStore

    from .document_store_manager import DocumentStoreContainer

    return DocumentStoreContainer(
        provider_name=DocumentStoreProvider.REDIS,
        store=RedisDocumentStore.from_host_and_port(
            host=cfg.pgvector_host,
            port=cfg.pgvector_port,
            namespace=table_name,
        ),
        table_name=table_name,
    )


def _pgvector(cfg: DocumentStoreConfig, table_name: str) -> DocumentStoreContainer:
    from llama_index.storage.docstore.postgres import PostgresDocumentStore

    from .document_store_manager import DocumentStoreContainer

    sec = cfg.pgvector_password
    if sec is None:
        raise ValueError("pgvector_password must be specified")

    return DocumentStoreContainer(
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
