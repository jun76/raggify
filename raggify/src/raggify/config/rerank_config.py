from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto

from mashumaro import DataClassDictMixin

__all__ = ["RerankProvider", "RerankConfig"]


class RerankProvider(StrEnum):
    FLAGEMBEDDING = auto()
    COHERE = auto()
    VOYAGE = auto()


@dataclass(kw_only=True)
class RerankConfig(DataClassDictMixin):
    """Config dataclass for rerank settings."""

    flagembedding_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    cohere_rerank_model: str = "rerank-multilingual-v3.0"
    voyage_rerank_model: str = "rerank-2.5"
    topk: int = 20
