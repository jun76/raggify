from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto


class RerankProvider(StrEnum):
    FLAGEMBEDDING = auto()
    COHERE = auto()


@dataclass(kw_only=True)
class RerankConfig:
    """リランク関連の設定用データクラス"""

    flagembedding_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    cohere_rerank_model: str = "rerank-multilingual-v3.0"
    topk: int = 20
