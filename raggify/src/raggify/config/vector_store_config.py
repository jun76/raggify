from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Optional

from mashumaro import DataClassDictMixin

from ..core.const import PROJECT_NAME

__all__ = ["VectorStoreProvider", "VectorStoreConfig"]


class VectorStoreProvider(StrEnum):
    PGVECTOR = auto()


@dataclass(kw_only=True)
class VectorStoreConfig(DataClassDictMixin):
    """Config dataclass for vector store settings."""

    # PGVector
    pgvector_host: str = "localhost"
    pgvector_port: int = 5432
    pgvector_database: str = PROJECT_NAME
    pgvector_user: str = PROJECT_NAME
    pgvector_password: Optional[str] = None
