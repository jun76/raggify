from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mashumaro import DataClassDictMixin

__all__ = ["GeneralConfig"]


@dataclass(kw_only=True)
class GeneralConfig(DataClassDictMixin):
    host: str = "localhost"
    port: int = 8000
    topk: int = 20
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
