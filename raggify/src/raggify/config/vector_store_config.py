from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Optional

from mashumaro import DataClassDictMixin

from ..core.const import DEFAULT_WORKSPACE_PATH, PROJECT_NAME

__all__ = ["VectorStoreProvider", "VectorStoreConfig"]


class VectorStoreProvider(StrEnum):
    CHROMA = auto()
    PGVECTOR = auto()
    REDIS = auto()


@dataclass(kw_only=True)
class VectorStoreConfig(DataClassDictMixin):
    """Config dataclass for vector store settings."""

    # Chroma
    chroma_persist_dir: Path = DEFAULT_WORKSPACE_PATH / "chroma_db"
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
