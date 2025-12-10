from __future__ import annotations

import os
from urllib.parse import urlparse

__all__ = ["Exts"]


class Exts:
    # For individual reference
    PNG: str = ".png"
    WAV: str = ".wav"
    PDF: str = ".pdf"
    HTML: str = ".html"
    SVG: str = ".svg"
    MP4: str = ".mp4"
    MP3: str = ".mp3"

    # Follow the extensions supported by reader
    # (llama_index.core.readers.file.base._try_loading_included_file_formats).
    # The reader also tries to load other extensions as text as a fallback,
    # so note that text extensions such as .txt are intentionally omitted.

    # Extensions that can be base64-encoded and passed to multimodal (image) embedding models
    IMAGE: set[str] = {PNG, ".jpg", ".jpeg", ".gif", ".webp"}

    # Extensions accepted by multimodal (audio) embedding models
    AUDIO: set[str] = {MP3, WAV, ".ogg"}

    # Extensions accepted by multimodal (video) embedding models
    VIDEO: set[str] = {
        MP4,
        ".mov",
        ".mkv",
        ".webm",
        ".flv",
        ".mpeg",
        ".mpg",
        ".wmv",
        ".3gp",
    }

    # Extensions used to detect sitemaps
    SITEMAP: set[str] = {".xml"}

    # Limit ingestion to avoid unexpected or huge media files
    DEFAULT_INGEST_TARGET: set[str] = {
        ".hwp",
        PDF,
        ".docx",
        ".pptx",
        ".ppt",
        ".pptm",
        ".csv",
        ".epub",
        ".mbox",
        ".ipynb",
        ".xls",
        ".xlsx",
        ".txt",
        ".text",
        ".md",
        ".json",
        HTML,
        ".tex",
    } | SITEMAP

    # Extensions to pass through without a dedicated reader so embedding models
    # handle them directly at upsert time
    PASS_THROUGH_MEDIA = AUDIO | VIDEO

    @classmethod
    def endswith_exts(cls, s: str, exts: set[str]) -> bool:
        """Return True if the string ends with any of the given extensions.

        Args:
            s (str): Input string.
            exts (set[str]): Extension set to check.

        Returns:
            bool: True when matched.
        """
        return any(s.lower().endswith(ext) for ext in exts)

    @classmethod
    def endswith_ext(cls, s: str, ext: str) -> bool:
        """Return True if the string ends with the given extension.

        Args:
            s (str): Input string.
            ext (str): Extension to check.

        Returns:
            bool: True when matched.
        """
        return s.lower().endswith(ext.lower())

    @classmethod
    def get_ext(cls, uri: str, dot: bool = True) -> str:
        """Get an extension from a file path or URL.

        Args:
            uri (str): File path or URL string.
            dot (bool, optional): Whether to include the dot. Defaults to True.

        Returns:
            str: Extension string.
        """
        parsed = urlparse(uri)
        ext = os.path.splitext(parsed.path)[1].lower()

        if not dot:
            return ext.replace(".", "")

        return ext
