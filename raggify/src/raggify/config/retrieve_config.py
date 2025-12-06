from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto

from mashumaro import DataClassDictMixin

__all__ = ["RetrieveMode", "RetrieveConfig"]


class RetrieveMode(StrEnum):
    VECTOR_ONLY = auto()
    BM25_ONLY = auto()
    FUSION = auto()


@dataclass(kw_only=True)
class RetrieveConfig(DataClassDictMixin):
    """Config dataclass for retriever settings."""

    mode: RetrieveMode = RetrieveMode.FUSION
    bm25_topk: int = 10
    fusion_lambda_vector: float = 0.5
    fusion_lambda_bm25: float = 0.5
