from __future__ import annotations

from dataclasses import dataclass

from .default_settings import DefaultSettings


@dataclass(kw_only=True)
class IngestCacheStoreConfig:
    """キャッシュストア関連の設定用データクラス"""

    # Redit
    redis_host: str = DefaultSettings.REDIS_HOST
    redis_port: int = DefaultSettings.REDIS_PORT

    # Local
    ingest_cache_local_persist_path: str = (
        DefaultSettings.INGEST_CACHE_LOCAL_PERSIST_PATH
    )
