from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .default_settings import DefaultSettings


@dataclass(kw_only=True)
class DocumentStoreConfig:
    """ドキュメントストア関連の設定用データクラス"""

    # PGVector
    pgvector_host: str = DefaultSettings.PGVECTOR_HOST
    pgvector_port: int = DefaultSettings.PGVECTOR_PORT
    pgvector_database: str = DefaultSettings.PGVECTOR_DATABASE
    pgvector_user: str = DefaultSettings.PGVECTOR_USER
    pgvector_password: Optional[str] = DefaultSettings.PGVECTOR_PASSWORD

    # Redit
    redis_host: str = DefaultSettings.REDIS_HOST
    redis_port: int = DefaultSettings.REDIS_PORT
