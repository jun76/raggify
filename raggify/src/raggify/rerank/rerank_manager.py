from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore


__all__ = ["RerankManager", "RerankContainer"]


@dataclass(kw_only=True)
class RerankContainer:
    """Aggregate parameters for reranking."""

    provider_name: str
    rerank: BaseNodePostprocessor


class RerankManager:
    """Manager class for reranking."""

    def __init__(self, cont: Optional[RerankContainer] = None) -> None:
        """Constructor.

        Args:
            cont (RerankContainer): Rerank container.
        """
        self._cont = cont

        if cont:
            logger.debug(f"{cont.provider_name} rerank initialized")
        else:
            logger.debug("rerank provider is not specified")

    @property
    def name(self) -> str:
        """Provider name.

        Returns:
            str: Provider name.
        """
        return self._cont.provider_name if self._cont else "none"

    async def arerank(
        self, nodes: list[NodeWithScore], query: str, topk: int
    ) -> list[NodeWithScore]:
        """Reorder results with a reranker based on the query.

        Args:
            nodes (list[NodeWithScore]): Nodes to rerank.
            query (str): Query string.
            topk (int): Number of items to retrieve.

        Returns:
            list[NodeWithScore]: Reranked nodes.

        Raises:
            RuntimeError: When reranking fails.
        """
        if self._cont is None:
            logger.info("rerank provider is not specified")
            return nodes

        # top_n should be set at instantiation,
        # but allow temporary changes for one-off retrievals by rewriting here.
        original_top_n: Optional[int] = None
        if hasattr(self._cont.rerank, "top_n"):
            original_top_n = getattr(self._cont.rerank, "top_n")
            setattr(self._cont.rerank, "top_n", topk)

        try:
            return await self._cont.rerank.apostprocess_nodes(
                nodes=nodes, query_str=query
            )
        except Exception as e:
            raise RuntimeError("failed to rerank documents") from e
        finally:
            if original_top_n:
                setattr(self._cont.rerank, "top_n", original_top_n)
