from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.config_manager import ConfigManager
from ..config.rerank_config import RerankProvider
from ..core.const import EXTRA_PKG_NOT_FOUND_MSG
from ..logger import logger

if TYPE_CHECKING:
    from .rerank_manager import RerankContainer, RerankManager


__all__ = ["create_rerank_manager"]


def create_rerank_manager(cfg: ConfigManager) -> RerankManager:
    """Create an instance of the rerank manager.

    Args:
        cfg (ConfigManager): Configuration manager.

    Raises:
        RuntimeError: Failed to create an instance.

    Returns:
        RerankManager: Rerank manager.
    """
    from .rerank_manager import RerankManager

    try:
        match cfg.general.rerank_provider:
            case RerankProvider.COHERE:
                rerank = _cohere(cfg)
            case RerankProvider.FLAGEMBEDDING:
                rerank = _flagembedding(cfg)
            case RerankProvider.VOYAGE:
                rerank = _voyage(cfg)
            case _:
                rerank = None

        return RerankManager(rerank)
    except Exception as e:
        raise RuntimeError(f"failed to create rerank: {e}") from e


# Container constructors for each provider.
def _cohere(cfg: ConfigManager) -> RerankContainer:
    try:
        from llama_index.postprocessor.cohere_rerank import CohereRerank  # type: ignore
    except ImportError:
        raise ImportError(
            EXTRA_PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-postprocessor-cohere-rerank",
                extra="rerank",
                feature="CohereRerank",
            )
        )

    from .rerank_manager import RerankContainer

    return RerankContainer(
        provider_name=RerankProvider.COHERE,
        rerank=CohereRerank(model=cfg.rerank.cohere_rerank_model),
    )


def _flagembedding(cfg: ConfigManager) -> RerankContainer:
    try:
        from llama_index.postprocessor.flag_embedding_reranker import (  # type: ignore
            FlagEmbeddingReranker,
        )
    except ImportError:
        raise ImportError(
            EXTRA_PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-postprocessor-flag-embedding-reranker",
                extra="rerank",
                feature="FlagEmbeddingReranker",
            )
        )

    from .rerank_manager import RerankContainer

    model = cfg.rerank.flagembedding_rerank_model
    logger.debug(f"Initializing FlagEmbedding Reranker with model: {model}")

    return RerankContainer(
        provider_name=RerankProvider.FLAGEMBEDDING,
        rerank=FlagEmbeddingReranker(model=model),
    )


def _voyage(cfg: ConfigManager) -> RerankContainer:
    try:
        from llama_index.postprocessor.voyageai_rerank import (  # type: ignore
            VoyageAIRerank,
        )
    except ImportError:
        raise ImportError(
            EXTRA_PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-postprocessor-voyageai-rerank",
                extra="rerank",
                feature="VoyageAIRerank",
            )
        )

    from .rerank_manager import RerankContainer

    return RerankContainer(
        provider_name=RerankProvider.VOYAGE,
        rerank=VoyageAIRerank(model=cfg.rerank.voyage_rerank_model),
    )
