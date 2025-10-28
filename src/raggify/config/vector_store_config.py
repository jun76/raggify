from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .default_settings import DefaultSettings


@dataclass(kw_only=True)
class VectorStoreConfig:
    """ベクトルストア関連の設定用データクラス"""

    # General
    cache_load_limit: int = DefaultSettings.CACHE_LOAD_LIMIT
    check_update: bool = DefaultSettings.CHECK_UPDATE

    # Chroma
    chroma_persist_dir: str = DefaultSettings.CHROMA_PERSIST_DIR
    chroma_host: Optional[str] = DefaultSettings.CHROMA_HOST
    chroma_port: Optional[int] = DefaultSettings.CHROMA_PORT
    chroma_tenant: Optional[str] = DefaultSettings.CHROMA_TENANT
    chroma_database: Optional[str] = DefaultSettings.CHROMA_DATABASE

    # PGVector
    pgvector_host: str = DefaultSettings.PGVECTOR_HOST
    pgvector_port: int = DefaultSettings.PGVECTOR_PORT
    pgvector_database: str = DefaultSettings.PGVECTOR_DATABASE
    pgvector_user: str = DefaultSettings.PGVECTOR_USER
    pgvector_password: Optional[str] = DefaultSettings.PGVECTOR_PASSWORD
