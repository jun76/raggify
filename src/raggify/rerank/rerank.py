from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.config_manager import ConfigManager
from ..config.rerank_config import RerankProvider

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
            case _:
                rerank = None

        return RerankManager(rerank)
    except Exception as e:
        raise RuntimeError(f"failed to create rerank: {e}") from e


# Container constructors for each provider.
def _cohere(cfg: ConfigManager) -> RerankContainer:
    from llama_index.postprocessor.cohere_rerank import CohereRerank

    from .rerank_manager import RerankContainer

    return RerankContainer(
        provider_name=RerankProvider.COHERE,
        rerank=CohereRerank(model=cfg.rerank.cohere_rerank_model),
    )


def _flagembedding(cfg: ConfigManager) -> RerankContainer:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

    from .rerank_manager import RerankContainer

    return RerankContainer(
        provider_name=RerankProvider.FLAGEMBEDDING,
        rerank=FlagEmbeddingReranker(model=cfg.rerank.flagembedding_rerank_model),
    )
