from __future__ import annotations

from dataclasses import dataclass

from .default_settings import DefaultSettings


@dataclass(kw_only=True)
class RerankConfig:
    """リランク関連の設定用データクラス"""

    flagembedding_rerank_model: str = DefaultSettings.FLAGEMBEDDING_RERANK_MODEL
    cohere_rerank_model: str = DefaultSettings.COHERE_RERANK_MODEL
    topk: int = DefaultSettings.TOPK
