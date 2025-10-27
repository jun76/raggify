from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from llama_index.core.schema import NodeWithScore

from ..config.default_settings import DefaultSettings as DS
from ..core.event import async_loop_runner
from ..llama.core.indices.multi_modal.retriever import AudioRetriever
from ..llama.core.schema import Modality
from ..logger import logger

if TYPE_CHECKING:
    from ..rerank.rerank_manager import RerankManager
    from ..runtime import Runtime
    from ..vector_store.vector_store_manager import VectorStoreManager

__all__ = [
    "query_text_text",
    "aquery_text_text",
    "query_text_image",
    "aquery_text_image",
    "query_image_image",
    "aquery_image_image",
    "query_text_audio",
    "aquery_text_audio",
    "query_audio_audio",
    "aquery_audio_audio",
]


def _rt() -> Runtime:
    """遅延ロード用ゲッター。

    Returns:
        Runtime: ランタイム
    """
    from ..runtime import get_runtime

    return get_runtime()


def query_text_text(
    query: str,
    store: Optional[VectorStoreManager] = None,
    topk: int = DS.TOPK,
    rerank: Optional[RerankManager] = None,
) -> list[NodeWithScore]:
    """クエリ文字列によるテキストドキュメント検索。

    Args:
        query (str): クエリ文字列
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        topk (int, optional): 取得件数。Defaults to DS.TOPK.
        rerank (Optional[RerankManager], optional): リランカー管理。Defaults to None.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    return async_loop_runner.run(
        lambda: aquery_text_text(query=query, store=store, topk=topk, rerank=rerank)
    )


async def aquery_text_text(
    query: str,
    store: Optional[VectorStoreManager] = None,
    topk: int = DS.TOPK,
    rerank: Optional[RerankManager] = None,
) -> list[NodeWithScore]:
    """クエリ文字列によるテキストドキュメント非同期検索。

    Args:
        query (str): クエリ文字列
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        topk (int, optional): 取得件数。Defaults to DS.TOPK.
        rerank (Optional[RerankManager], optional): リランカー管理。Defaults to None.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    store = store or _rt().vector_store

    index = store.get_index(Modality.TEXT)
    if index is None:
        logger.error("store is not initialized")
        return []

    retriever_engine = index.as_retriever(similarity_top_k=topk)
    nwss = await retriever_engine.aretrieve(query)

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    rerank = rerank or _rt().rerank_manager
    if rerank is None:
        return nwss

    nwss = await rerank.arerank(nodes=nwss, query=query)
    logger.debug(f"reranked {len(nwss)} nodes")

    return nwss


def query_text_image(
    query: str,
    store: Optional[VectorStoreManager] = None,
    topk: int = DS.TOPK,
    rerank: Optional[RerankManager] = None,
) -> list[NodeWithScore]:
    """クエリ文字列による画像ドキュメント検索。

    Args:
        query (str): クエリ文字列
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        topk (int, optional): 取得件数。Defaults to DS.TOPK.
        rerank (Optional[RerankManager], optional): リランカー管理。Defaults to None.

    Raises:
        RuntimeError: テキスト --> 画像埋め込み非対応

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    return async_loop_runner.run(
        lambda: aquery_text_image(query=query, store=store, topk=topk, rerank=rerank)
    )


async def aquery_text_image(
    query: str,
    store: Optional[VectorStoreManager] = None,
    topk: int = DS.TOPK,
    rerank: Optional[RerankManager] = None,
) -> list[NodeWithScore]:
    """クエリ文字列による画像ドキュメント非同期検索。

    Args:
        query (str): クエリ文字列
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        topk (int, optional): 取得件数。Defaults to DS.TOPK.
        rerank (Optional[RerankManager], optional): リランカー管理。Defaults to None.

    Raises:
        RuntimeError: テキスト --> 画像埋め込み非対応

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex

    store = store or _rt().vector_store

    index = store.get_index(Modality.IMAGE)
    if index is None:
        logger.error("store is not initialized")
        return []

    if not isinstance(index, MultiModalVectorStoreIndex):
        logger.error("multimodal index is required")
        return []

    retriever_engine = index.as_retriever(
        similarity_top_k=topk, image_similarity_top_k=topk
    )

    try:
        nwss = await retriever_engine.atext_to_image_retrieve(query)
    except Exception as e:
        raise RuntimeError(
            "this embed model may not support text --> image embedding"
        ) from e

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    rerank = rerank or _rt().rerank_manager
    if rerank is None:
        return nwss

    nwss = await rerank.arerank(nodes=nwss, query=query)
    logger.debug(f"reranked {len(nwss)} nodes")

    return nwss


def query_image_image(
    path: str,
    store: Optional[VectorStoreManager] = None,
    topk: int = DS.TOPK,
) -> list[NodeWithScore]:
    """クエリ画像による画像ドキュメント検索。

    Args:
        path (str): クエリ画像の ローカルパス
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        topk (int, optional): 取得件数。Defaults to DS.TOPK.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    return async_loop_runner.run(
        lambda: aquery_image_image(path=path, store=store, topk=topk)
    )


async def aquery_image_image(
    path: str,
    store: Optional[VectorStoreManager] = None,
    topk: int = DS.TOPK,
) -> list[NodeWithScore]:
    """クエリ画像による画像ドキュメント非同期検索。

    Args:
        path (str): クエリ画像の ローカルパス
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        topk (int, optional): 取得件数。Defaults to DS.TOPK.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex

    store = store or _rt().vector_store

    index = store.get_index(Modality.IMAGE)
    if index is None:
        logger.error("store is not initialized")
        return []

    if not isinstance(index, MultiModalVectorStoreIndex):
        logger.error("multimodal index is required")
        return []

    retriever_engine = index.as_retriever(
        similarity_top_k=topk, image_similarity_top_k=topk
    )
    nwss = await retriever_engine.aimage_to_image_retrieve(path)

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    logger.debug(f"got {len(nwss)} nodes")

    return nwss


def query_text_audio(
    query: str,
    store: Optional[VectorStoreManager] = None,
    topk: int = DS.TOPK,
    rerank: Optional[RerankManager] = None,
) -> list[NodeWithScore]:
    """クエリ文字列による音声ドキュメント検索。

    Args:
        query (str): クエリ文字列
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        topk (int, optional): 取得件数。Defaults to DS.TOPK.
        rerank (Optional[RerankManager], optional): リランカー管理。Defaults to None.

    Raises:
        RuntimeError: テキスト --> 音声埋め込み非対応

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    return async_loop_runner.run(
        lambda: aquery_text_audio(query=query, store=store, topk=topk, rerank=rerank)
    )


async def aquery_text_audio(
    query: str,
    store: Optional[VectorStoreManager] = None,
    topk: int = DS.TOPK,
    rerank: Optional[RerankManager] = None,
) -> list[NodeWithScore]:
    """クエリ文字列による音声ドキュメント非同期検索。

    Args:
        query (str): クエリ文字列
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        topk (int, optional): 取得件数。Defaults to DS.TOPK.
        rerank (Optional[RerankManager], optional): リランカー管理。Defaults to None.

    Raises:
        RuntimeError: テキスト --> 音声埋め込み非対応

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    store = store or _rt().vector_store

    index = store.get_index(Modality.AUDIO)
    if index is None:
        logger.error("store is not initialized")
        return []

    retriever_engine = AudioRetriever(index=index, top_k=topk)
    try:
        nwss = await retriever_engine.atext_to_audio_retrieve(query)
    except Exception as e:
        raise RuntimeError(
            "this embed model may not support text --> audio embedding"
        ) from e

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    rerank = rerank or _rt().rerank_manager
    if rerank is None:
        return nwss

    nwss = await rerank.arerank(nodes=nwss, query=query)
    logger.debug(f"reranked {len(nwss)} nodes")

    return nwss


def query_audio_audio(
    path: str,
    store: Optional[VectorStoreManager] = None,
    topk: int = DS.TOPK,
) -> list[NodeWithScore]:
    """クエリ音声による音声ドキュメント検索。

    Args:
        path (str): クエリ音声の ローカルパス
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        topk (int, optional): 取得件数。Defaults to DS.TOPK.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    return async_loop_runner.run(
        lambda: aquery_audio_audio(path=path, store=store, topk=topk)
    )


async def aquery_audio_audio(
    path: str,
    store: Optional[VectorStoreManager] = None,
    topk: int = DS.TOPK,
) -> list[NodeWithScore]:
    """クエリ音声による音声ドキュメント非同期検索。

    Args:
        path (str): クエリ音声の ローカルパス
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        topk (int, optional): 取得件数。Defaults to DS.TOPK.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    store = store or _rt().vector_store

    index = store.get_index(Modality.AUDIO)
    if index is None:
        logger.error("store is not initialized")
        return []

    retriever_engine = AudioRetriever(index=index, top_k=topk)
    nwss = await retriever_engine.aaudio_to_audio_retrieve(path)

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    logger.debug(f"got {len(nwss)} nodes")

    return nwss
