from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mashumaro import DataClassDictMixin

from ..core.const import (
    DEFAULT_KNOWLEDGEBASE_NAME,
    DEFAULT_WORKSPACE_PATH,
    PROJECT_NAME,
)


@dataclass(kw_only=True)
class IngestConfig(DataClassDictMixin):
    """Config dataclass for document ingestion settings."""

    # General
    chunk_size: int = 500
    chunk_overlap: int = 50
    upload_dir: Path = DEFAULT_WORKSPACE_PATH / "upload"
    pipe_persist_dir: Path = DEFAULT_WORKSPACE_PATH / DEFAULT_KNOWLEDGEBASE_NAME
    batch_size: int = 100
    audio_chunk_seconds: Optional[int] = 25
    video_chunk_seconds: Optional[int] = 25
    additional_exts: set[str] = field(default_factory=lambda: {".c", ".py", ".rst"})

    # Web
    user_agent: str = PROJECT_NAME
    load_asset: bool = True
    req_per_sec: int = 2
    timeout_sec: int = 30
    same_origin: bool = True
    max_asset_bytes: int = 100 * 1024 * 1024  # 100 MB
