from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import cfg
from ..config.default_settings import VectorStoreProvider
from ..llama.core.schema import Modality

if TYPE_CHECKING:
    from ..embed.embed_manager import EmbedManager
    from ..meta_store.structured.structured import Structured
    from .vector_store_manager import VectorStoreContainer, VectorStoreManager

__all__ = ["create_vector_store_manager"]


def create_vector_store_manager(
    embed: EmbedManager,
    meta_store: Structured,
) -> VectorStoreManager:
    """ベクトルストア管理のインスタンスを生成する。

    Args:
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
            conts[Modality.TEXT] = _create_container(embed.space_key_text)

        if cfg.general.image_embed_provider:
            conts[Modality.IMAGE] = _create_container(embed.space_key_image)

        if cfg.general.audio_embed_provider:
            conts[Modality.AUDIO] = _create_container(embed.space_key_audio)
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


def _create_container(space_key: str) -> VectorStoreContainer:
    """空間キー毎のベクトルストアコンテナを生成する。

    Args:
        space_key (str): 空間キー

    Raises:
        RuntimeError: サポート外のプロバイダ

    Returns:
        VectorStoreContainer: コンテナ
    """
    table_name = _generate_table_name(space_key)
    match cfg.general.vector_store_provider:
        case VectorStoreProvider.PGVECTOR:
            cont = _pgvector(table_name)
        case VectorStoreProvider.CHROMA:
            cont = _chroma(table_name)
        case _:
            raise RuntimeError(
                f"unsupported vector store: {cfg.general.vector_store_provider}"
            )

    return cont


def _generate_table_name(space_key: str) -> str:
    """テーブル名を生成する。

    Args:
        space_key (str): 空間キー

    Returns:
        str: テーブル名
    """
    return f"{cfg.project_name}__{cfg.general.knowledgebase_name}__{space_key}"


# 以下、プロバイダ毎のコンテナ生成ヘルパー
def _pgvector(table_name: str) -> VectorStoreContainer:
    """ベクトルストアコンテナ生成ヘルパー

    Args:
        table_name (str): テーブル名

    Raise:
        ValueError: パスワード未指定

    Returns:
        VectorStoreContainer: コンテナ
    """
    from llama_index.vector_stores.postgres import PGVectorStore

    from .vector_store_manager import VectorStoreContainer

    sec = cfg.vector_store.pgvector_password
    if sec is None:
        raise ValueError("pgvector_password must be specified")

    return VectorStoreContainer(
        provider_name=VectorStoreProvider.PGVECTOR,
        store=PGVectorStore.from_params(
            host=cfg.vector_store.pgvector_host,
            port=str(cfg.vector_store.pgvector_port),
            database=cfg.vector_store.pgvector_database,
            user=cfg.vector_store.pgvector_user,
            password=cfg.vector_store.pgvector_password,
            table_name=table_name,
        ),
        table_name=table_name,
    )


def _chroma(table_name: str) -> VectorStoreContainer:
    """ベクトルストアコンテナ生成ヘルパー

    Args:
        table_name (str): テーブル名

    Raises:
        RuntimeError: パラメータ指定漏れ

    Returns:
        VectorStoreContainer: コンテナ
    """
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    from .vector_store_manager import VectorStoreContainer

    if (
        cfg.vector_store.chroma_host is not None
        and cfg.vector_store.chroma_port is not None
    ):
        client = chromadb.HttpClient(
            host=cfg.vector_store.chroma_host,
            port=cfg.vector_store.chroma_port,
        )
    elif cfg.vector_store.chroma_persist_dir:
        client = chromadb.PersistentClient(path=cfg.vector_store.chroma_persist_dir)
    else:
        raise RuntimeError("persist_directory or host + port must be specified")

    collection = client.get_or_create_collection(table_name)

    return VectorStoreContainer(
        provider_name=VectorStoreProvider.CHROMA,
        store=ChromaVectorStore(chroma_collection=collection),
        table_name=table_name,
    )
