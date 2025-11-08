from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Optional

from ..core.const import PROJECT_NAME


class DocumentStoreProvider(StrEnum):
    REDIS = auto()
    POSTGRES = auto()
    LOCAL = auto()


@dataclass(kw_only=True)
class DocumentStoreConfig:
    """ドキュメントストア関連の設定用データクラス"""

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = PROJECT_NAME
    postgres_user: str = PROJECT_NAME
    postgres_password: Optional[str] = None
