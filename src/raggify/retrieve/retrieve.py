from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from llama_index.core.async_utils import asyncio_run
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore
from llama_index.retrievers.bm25 import BM25Retriever

from ..config.retrieve_config import RetrieveMode
from ..llama.core.indices.multi_modal.retriever import AudioRetriever
from ..llama.core.schema import Modality
from ..logger import logger
from ..runtime import get_runtime as _rt

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


def _get_vector_retriever(rt: Runtime, index: VectorStoreIndex) -> BaseRetriever:
    """ベクトル検索用リトリーバーを取得する。

    Args:
        rt (Runtime): ランタイム
        index (VectorStoreIndex): インデックス

    Returns:
        BaseRetriever: リトリーバー
    """
    return index.as_retriever(similarity_top_k=rt.cfg.rerank.topk)


def _get_bm25_retriever(rt: Runtime) -> Optional[BaseRetriever]:
    """BM25 モード検索用リトリーバーを取得する。

    コーパスが無い場合は不発。

    Args:
        rt (Runtime): ランタイム

    Returns:
        Optional[BaseRetriever]: リトリーバー
    """
    docstore = rt.document_store
    if not docstore.has_bm25_corpus():
        logger.warning("docstore is empty; BM25 retrieval skipped")
        return None

    try:
        return BM25Retriever.from_defaults(
            docstore=docstore.store,
            similarity_top_k=rt.cfg.retrieve.bm25_topk,
        )
    except Exception as e:
        logger.warning(f"failed to get BM25 retriever: {e}")
        return None


def _get_fusion_retriever(rt: Runtime, index: VectorStoreIndex) -> BaseRetriever:
    """ベクトルと BM25 のフュージョン検索用リトリーバーを取得する。

    コーパスが無い場合はベクトル検索にフォールバック。

    Args:
        rt (Runtime): ランタイム
        index (VectorStoreIndex): インデックス

    Returns:
        BaseRetriever: リトリーバー
    """
    docstore = rt.document_store
    topk = rt.cfg.rerank.topk

    if not docstore.has_bm25_corpus():
        logger.warning("docstore is empty; falling back to vector-only retrieval")
        return index.as_retriever(similarity_top_k=topk)

    vector_retriever = index.as_retriever(similarity_top_k=topk)

    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore.store,
        similarity_top_k=rt.cfg.retrieve.bm25_topk,
    )

    return QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=topk,
        num_queries=1,
        mode=FUSION_MODES.RELATIVE_SCORE,
        retriever_weights=[
            rt.cfg.retrieve.fusion_lambda_vector,
            rt.cfg.retrieve.fusion_lambda_bm25,
        ],
        verbose=False,
    )


def query_text_text(
    query: str,
    topk: Optional[int] = None,
    mode: Optional[RetrieveMode] = None,
) -> list[NodeWithScore]:
    """クエリ文字列によるテキストドキュメント検索。

    Args:
        query (str): クエリ文字列
        topk (int, optional): 取得件数。Defaults to None.
        mode (Optional[RetrieveMode], optional): BM25 ハイブリッド検索を使用するか。Defaults to None.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    topk = topk or _rt().cfg.rerank.topk

    return asyncio_run(aquery_text_text(query=query, topk=topk, mode=mode))


async def aquery_text_text(
    query: str,
    topk: Optional[int] = None,
    mode: Optional[RetrieveMode] = None,
) -> list[NodeWithScore]:
    """クエリ文字列によるテキストドキュメント非同期検索。

    Args:
        query (str): クエリ文字列
        topk (int, optional): 取得件数。Defaults to None.
        mode (Optional[RetrieveMode], optional): 検索モード。Defaults to None.

    Returns:
        list[NodeWithScore]: 検索結果のリスト
    """
    rt = _rt()
    store = rt.vector_store
    index = store.get_index(Modality.TEXT)
    if index is None:
        logger.error("store is not initialized")
        return []

    mode = mode or rt.cfg.retrieve.mode

    match mode:
        case RetrieveMode.VECTOR_ONLY:
            retriever_engine = _get_vector_retriever(rt=rt, index=index)
        case RetrieveMode.BM25_ONLY:
            retriever_engine = _get_bm25_retriever(rt)
        case RetrieveMode.FUSION:
            retriever_engine = _get_fusion_retriever(rt=rt, index=index)
        case _:
            raise ValueError(f"unexpected retrieve mode: {mode}")

    if retriever_engine is None:
        return []

    nwss = await retriever_engine.aretrieve(query)
    if len(nwss) == 0:
        logger.warning("empty nodes")
        return []

    rerank = rt.rerank_manager
    if rerank is None:
        return nwss

    topk = topk or rt.cfg.rerank.topk
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
