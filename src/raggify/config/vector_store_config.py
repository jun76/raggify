from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Optional

from ..core.const import PROJECT_NAME


class VectorStoreProvider(StrEnum):
    CHROMA = auto()
    PGVECTOR = auto()
    REDIS = auto()


@dataclass(kw_only=True)
class VectorStoreConfig:
    """ベクトルストア関連の設定用データクラス"""

    # General
    cache_load_limit: int = 10000
    check_update: bool = False

    # Chroma
    chroma_persist_dir: str = f"/etc/{PROJECT_NAME}/{PROJECT_NAME}_db"
    chroma_host: Optional[str] = None
    chroma_port: Optional[int] = None
    chroma_tenant: Optional[str] = None
    chroma_database: Optional[str] = None

    # PGVector
    pgvector_host: str = "localhost"
    pgvector_port: int = 5432
    pgvector_database: str = PROJECT_NAME
    pgvector_user: str = PROJECT_NAME
    pgvector_password: Optional[str] = None

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
