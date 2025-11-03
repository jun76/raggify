from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto


class DocumentStoreProvider(StrEnum):
    REDIS = auto()
    LOCAL = auto()


@dataclass(kw_only=True)
class DocumentStoreConfig:
    """ドキュメントストア関連の設定用データクラス"""

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
