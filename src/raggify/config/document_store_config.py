from __future__ import annotations

from dataclasses import dataclass

from .default_settings import DefaultSettings


@dataclass(kw_only=True)
class DocumentStoreConfig:
    """ドキュメントストア関連の設定用データクラス"""

    # Redit
    redis_host: str = DefaultSettings.REDIS_HOST
    redis_port: int = DefaultSettings.REDIS_PORT
