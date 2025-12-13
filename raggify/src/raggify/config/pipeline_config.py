from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from mashumaro import DataClassDictMixin

from ..core.const import DEFAULT_KNOWLEDGEBASE_NAME, DEFAULT_WORKSPACE_PATH

__all__ = ["PipelineConfig"]


@dataclass(kw_only=True)
class PipelineConfig(DataClassDictMixin):
    """Config dataclass for ingest pipeline settings."""

    persist_dir: Path = DEFAULT_WORKSPACE_PATH / DEFAULT_KNOWLEDGEBASE_NAME
    batch_size: int = 10
    batch_interval_sec: float = 0.5
    batch_retry_interval_sec: list[float] = field(
        default_factory=lambda: [1.0, 2.0, 4.0, 8.0, 16.0]
    )
