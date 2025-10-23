from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import cfg
from ..config.default_settings import RerankProvider

if TYPE_CHECKING:
    from .rerank_manager import RerankContainer, RerankManager


__all__ = ["create_rerank_manager"]


def create_rerank_manager() -> RerankManager:
    """リランク管理のインスタンスを生成する。

    Raises:
        RuntimeError: インスタンス生成に失敗

    Returns:
        RerankManager: リランク管理
    """
    from .rerank_manager import RerankManager

    try:
        match cfg.general.rerank_provider:
            case RerankProvider.COHERE:
                rerank = _cohere()
            case RerankProvider.FLAGEMBEDDING:
                rerank = _flagembedding()
            case _:
                rerank = None

        return RerankManager(rerank)
    except Exception as e:
        raise RuntimeError(f"failed to create rerank: {e}") from e


def _cohere() -> RerankContainer:
    """リランク管理生成ヘルパー

    Returns:
        RerankContainer: コンテナ
    """
    from llama_index.postprocessor.cohere_rerank import CohereRerank

    from .rerank_manager import RerankContainer

    return RerankContainer(
        provider_name=RerankProvider.COHERE,
        rerank=CohereRerank(
            model=cfg.rerank.cohere_rerank_model, top_n=cfg.rerank.topk
        ),
    )


def _flagembedding() -> RerankContainer:
    """リランク管理生成ヘルパー

    Returns:
        RerankContainer: コンテナ
    """
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

    from .rerank_manager import RerankContainer

    return RerankContainer(
        provider_name=RerankProvider.FLAGEMBEDDING,
        rerank=FlagEmbeddingReranker(
            model=cfg.rerank.flagembedding_rerank_model, top_n=cfg.rerank.topk
        ),
    )
