from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import Optional

from mashumaro import DataClassDictMixin

from ..core.const import DEFAULT_WORKSPACE_PATH, PROJECT_NAME

__all__ = ["ParserProvider", "IngestConfig"]


class ParserProvider(StrEnum):
    LOCAL = auto()
    LLAMA_CLOUD = auto()


@dataclass(kw_only=True)
class IngestConfig(DataClassDictMixin):
    """Config dataclass for document ingestion settings."""

    # General
    text_primary_chunk_size: int = 2000
    text_chunk_size: int = 500
    text_chunk_overlap: int = 50
    upload_dir: Path = DEFAULT_WORKSPACE_PATH / "upload"
    audio_chunk_seconds: Optional[int] = 25
    video_chunk_seconds: Optional[int] = 25
    additional_exts: set[str] = field(default_factory=lambda: {".c", ".py", ".rst"})
    skip_known_sources: bool = False

    # Web
    user_agent: str = PROJECT_NAME
    load_asset: bool = True
    req_per_sec: int = 2
    timeout_sec: int = 30
    same_origin: bool = True
    max_asset_bytes: int = 100 * 1024 * 1024  # 100 MB
    include_selectors: list[str] = field(
        default_factory=lambda: [
            "article",
            "main",
            "body",
            '[role="main"]',
            "div#content",
            "div.content",
            ".entry-content",
            ".post",
        ]
    )
    exclude_selectors: list[str] = field(
        default_factory=lambda: [
            "nav",
            "footer",
            "aside",
            "header",
            ".ads",
            ".advert",
            ".share",
            ".breadcrumb",
            ".toc",
            ".related",
            ".sidebar",
        ]
    )
    strip_tags: list[str] = field(
        default_factory=lambda: [
            "script",
            "style",
            "noscript",
            "iframe",
            "form",
            "button",
            "input",
            "svg",
            "ins",
        ]
    )
    strip_query_keys: list[str] = field(
        default_factory=lambda: [
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_term",
            "utm_content",
            "gclid",
            "fbclid",
            "nocache",
        ]
    )
