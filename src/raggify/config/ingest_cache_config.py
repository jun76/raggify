from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto


class IngestCacheStoreProvider(StrEnum):
    REDIS = auto()
    LOCAL = auto()


@dataclass(kw_only=True)
class IngestCacheConfig:
    """インジェストキャッシュ関連の設定用データクラス"""

    # Redit
    redis_host: str = "localhost"
    redis_port: int = 6379
