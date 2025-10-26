from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.config_manager import ConfigManager
from ..config.default_settings import VectorStoreProvider
from ..config.vector_store_config import VectorStoreConfig
from ..llama.core.schema import Modality

if TYPE_CHECKING:
    from ..embed.embed_manager import EmbedManager
    from ..meta_store.structured.structured import Structured
    from .vector_store_manager import VectorStoreContainer, VectorStoreManager

__all__ = ["create_vector_store_manager"]


def create_vector_store_manager(
    cfg: ConfigManager,
    embed: EmbedManager,
    meta_store: Structured,
) -> VectorStoreManager:
    """ベクトルストア管理のインスタンスを生成する。

    Args:
        cfg (ConfigManager): 設定管理
        embed (EmbedManager): 埋め込み管理
        meta_store (Structured): メタデータ管理

    Raises:
        RuntimeError: インスタンス生成に失敗またはプロバイダ指定漏れ

    Returns:
        VectorStoreManager: ベクトルストア管理
    """
    from .vector_store_manager import VectorStoreManager

    try:
        conts: dict[Modality, VectorStoreContainer] = {}
        if cfg.general.text_embed_provider:
            conts[Modality.TEXT] = _create_container(
                cfg=cfg,
                space_key=embed.space_key_text,
                dim=embed.get_container(Modality.TEXT).dim,
            )

        if cfg.general.image_embed_provider:
            conts[Modality.IMAGE] = _create_container(
                cfg=cfg,
                space_key=embed.space_key_image,
                dim=embed.get_container(Modality.IMAGE).dim,
            )

        if cfg.general.audio_embed_provider:
            conts[Modality.AUDIO] = _create_container(
                cfg=cfg,
                space_key=embed.space_key_audio,
                dim=embed.get_container(Modality.AUDIO).dim,
            )
    except Exception as e:
        raise RuntimeError(f"failed to create vector store: {e}") from e

    if not conts:
        raise RuntimeError("no embedding providers are specified")

    return VectorStoreManager(
        conts=conts,
        embed=embed,
        meta_store=meta_store,
        cache_load_limit=cfg.vector_store.cache_load_limit,
        check_update=cfg.vector_store.check_update,
    )


def _create_container(
    cfg: ConfigManager, space_key: str, dim: int
) -> VectorStoreContainer:
    """空間キー毎のベクトルストアコンテナを生成する。

    Args:
        cfg (ConfigManager): 設定管理
        space_key (str): 空間キー
        dim (int): 埋め込み次元

    Raises:
        RuntimeError: サポート外のプロバイダ

    Returns:
        VectorStoreContainer: コンテナ
    """
    table_name = _generate_table_name(cfg, space_key)
    match cfg.general.vector_store_provider:
        case VectorStoreProvider.PGVECTOR:
            cont = _pgvector(cfg=cfg.vector_store, table_name=table_name, dim=dim)
        case VectorStoreProvider.CHROMA:
            cont = _chroma(cfg=cfg.vector_store, table_name=table_name, dim=dim)
        case _:
            raise RuntimeError(
                f"unsupported vector store: {cfg.general.vector_store_provider}"
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
def _pgvector(
    cfg: VectorStoreConfig, table_name: str, dim: int
) -> VectorStoreContainer:
    from llama_index.vector_stores.postgres import PGVectorStore

    from .vector_store_manager import VectorStoreContainer

    sec = cfg.pgvector_password
    if sec is None:
        raise ValueError("pgvector_password must be specified")

    return VectorStoreContainer(
        provider_name=VectorStoreProvider.PGVECTOR,
        store=PGVectorStore.from_params(
            host=cfg.pgvector_host,
            port=str(cfg.pgvector_port),
            database=cfg.pgvector_database,
            user=cfg.pgvector_user,
            password=sec,
            table_name=table_name,
            embed_dim=dim,
        ),
        table_name=table_name,
    )


def _chroma(cfg: VectorStoreConfig, table_name: str, dim: int) -> VectorStoreContainer:
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    from .vector_store_manager import VectorStoreContainer

    if cfg.chroma_host is not None and cfg.chroma_port is not None:
        client = chromadb.HttpClient(
            host=cfg.chroma_host,
            port=cfg.chroma_port,
        )
    elif cfg.chroma_persist_dir:
        client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
    else:
        raise RuntimeError("persist_directory or host + port must be specified")

    collection = client.get_or_create_collection(table_name)

    return VectorStoreContainer(
        provider_name=VectorStoreProvider.CHROMA,
        store=ChromaVectorStore(chroma_collection=collection),
        table_name=table_name,
    )
