from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Optional

from mashumaro import DataClassDictMixin

from ..core.const import PROJECT_NAME

__all__ = ["IngestCacheProvider", "IngestCacheConfig"]


class IngestCacheProvider(StrEnum):
    POSTGRES = auto()


@dataclass(kw_only=True)
class IngestCacheConfig(DataClassDictMixin):
    """Config dataclass for ingest cache settings."""

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = PROJECT_NAME
    postgres_user: str = PROJECT_NAME
    postgres_password: Optional[str] = None
