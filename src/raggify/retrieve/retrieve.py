from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from llama_index.core.async_utils import asyncio_run
from llama_index.core.schema import NodeWithScore

from ..llama.core.indices.multi_modal.retriever import AudioRetriever
from ..llama.core.schema import Modality
from ..logger import logger

if TYPE_CHECKING:
    from ..runtime import Runtime

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

_TOPK = 10


def _rt() -> Runtime:
    """遅延ロード用ゲッター。

    Returns:
        Runtime: ランタイム
    """
    from ..runtime import get_runtime

    return get_runtime()


def query_text_text(
    query: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    """クエリ文字列によるテキストドキュメント検索。

    Args:
        query (str): クエリ文字列
        topk (int, optional): 取得件数。Defaults to None.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    topk = topk or _rt().cfg.rerank.topk

    return asyncio_run(aquery_text_text(query=query, topk=topk))


async def aquery_text_text(
    query: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    """クエリ文字列によるテキストドキュメント非同期検索。

    Args:
        query (str): クエリ文字列
        topk (int, optional): 取得件数。Defaults to None.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    rt = _rt()
    store = rt.vector_store
    index = store.get_index(Modality.TEXT)
    if index is None:
        logger.error("store is not initialized")
        return []

    topk = topk or rt.cfg.rerank.topk
    retriever_engine = index.as_retriever(similarity_top_k=topk)
    nwss = await retriever_engine.aretrieve(query)

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    rerank = rt.rerank_manager
    if rerank is None:
        return nwss

    nwss = await rerank.arerank(nodes=nwss, query=query, topk=topk)
    logger.debug(f"reranked {len(nwss)} nodes")

    return nwss


def query_text_image(
    query: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    """クエリ文字列による画像ドキュメント検索。

    Args:
        query (str): クエリ文字列
        topk (int, optional): 取得件数。Defaults to None.

    Raises:
        RuntimeError: テキスト --> 画像埋め込み非対応

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    topk = topk or _rt().cfg.rerank.topk

    return asyncio_run(aquery_text_image(query=query, topk=topk))


async def aquery_text_image(
    query: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    """クエリ文字列による画像ドキュメント非同期検索。

    Args:
        query (str): クエリ文字列
        topk (int, optional): 取得件数。Defaults to None.

    Raises:
        RuntimeError: テキスト --> 画像埋め込み非対応

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex

    rt = _rt()
    store = rt.vector_store
    index = store.get_index(Modality.IMAGE)
    if index is None:
        logger.error("store is not initialized")
        return []

    if not isinstance(index, MultiModalVectorStoreIndex):
        logger.error("multimodal index is required")
        return []

    topk = topk or rt.cfg.rerank.topk
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

    rerank = rt.rerank_manager
    if rerank is None:
        return nwss

    nwss = await rerank.arerank(nodes=nwss, query=query, topk=topk)
    logger.debug(f"reranked {len(nwss)} nodes")

    return nwss


def query_image_image(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    """クエリ画像による画像ドキュメント検索。

    Args:
        path (str): クエリ画像の ローカルパス
        topk (int, optional): 取得件数。Defaults to None.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    topk = topk or _rt().cfg.rerank.topk

    return asyncio_run(aquery_image_image(path=path, topk=topk))


async def aquery_image_image(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    """クエリ画像による画像ドキュメント非同期検索。

    Args:
        path (str): クエリ画像の ローカルパス
        topk (int, optional): 取得件数。Defaults to None.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    from llama_index.core.indices.multi_modal import MultiModalVectorStoreIndex

    rt = _rt()
    store = rt.vector_store
    index = store.get_index(Modality.IMAGE)
    if index is None:
        logger.error("store is not initialized")
        return []

    if not isinstance(index, MultiModalVectorStoreIndex):
        logger.error("multimodal index is required")
        return []

    topk = topk or rt.cfg.rerank.topk
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
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    """クエリ文字列による音声ドキュメント検索。

    Args:
        query (str): クエリ文字列
        topk (int, optional): 取得件数。Defaults to None.

    Raises:
        RuntimeError: テキスト --> 音声埋め込み非対応

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    topk = topk or _rt().cfg.rerank.topk

    return asyncio_run(aquery_text_audio(query=query, topk=topk))


async def aquery_text_audio(
    query: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    """クエリ文字列による音声ドキュメント非同期検索。

    Args:
        query (str): クエリ文字列
        topk (int, optional): 取得件数。Defaults to None.

    Raises:
        RuntimeError: テキスト --> 音声埋め込み非対応

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    rt = _rt()
    store = rt.vector_store
    index = store.get_index(Modality.AUDIO)
    if index is None:
        logger.error("store is not initialized")
        return []

    topk = topk or rt.cfg.rerank.topk
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

    rerank = rt.rerank_manager
    if rerank is None:
        return nwss

    nwss = await rerank.arerank(nodes=nwss, query=query, topk=topk)
    logger.debug(f"reranked {len(nwss)} nodes")

    return nwss


def query_audio_audio(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    """クエリ音声による音声ドキュメント検索。

    Args:
        path (str): クエリ音声の ローカルパス
        topk (int, optional): 取得件数。Defaults to None.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    topk = topk or _rt().cfg.rerank.topk

    return asyncio_run(aquery_audio_audio(path=path, topk=topk))


async def aquery_audio_audio(
    path: str,
    topk: Optional[int] = None,
) -> list[NodeWithScore]:
    """クエリ音声による音声ドキュメント非同期検索。

    Args:
        path (str): クエリ音声の ローカルパス
        topk (int, optional): 取得件数。Defaults to None.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    rt = _rt()
    store = rt.vector_store
    index = store.get_index(Modality.AUDIO)
    if index is None:
        logger.error("store is not initialized")
        return []

    topk = topk or rt.cfg.rerank.topk
    retriever_engine = AudioRetriever(index=index, top_k=topk)
    nwss = await retriever_engine.aaudio_to_audio_retrieve(path)

    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    logger.debug(f"got {len(nwss)} nodes")

    return nwss
