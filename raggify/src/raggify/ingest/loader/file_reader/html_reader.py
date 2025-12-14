from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Iterable

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ....logger import logger

if TYPE_CHECKING:
    from ....config.ingest_config import IngestConfig

__all__ = ["HTMLReader"]


class HTMLReader(BaseReader):
    """HTML file reader that extracts text content from HTML files.

    Note:
        Using SimpleWebPageReader causes two fetches to run during asset collection,
        so we reuse the HTML files saved to temporary files by this custom Reader.
    """

    def __init__(self, cfg: IngestConfig) -> None:
        """Constructor.

        Args:
            cfg (IngestConfig): Ingest configuration.
        """
        super().__init__()
        self._cfg = cfg

    def lazy_load_data(self, path: Any, extra_info: Any = None) -> Iterable[Document]:
        """Load an HTML file and generate text documents.

        Args:
            path (Any): File path-like object.

        Returns:
            Iterable[Document]: List of documents read from the HTML file.
        """
        from ....core.metadata import BasicMetaData

        try:
            path = os.fspath(path)
            with open(path, "r", encoding="utf-8") as f:
                import html2text

                html = self._cleanse_html_text(f.read())

                # Convert to markdown-like text
                text = html2text.html2text(html)
        except OSError as e:
            logger.warning(f"failed to read HTML file {path}: {e}")
            return []

        metadata = BasicMetaData()
        doc = Document(text=text, metadata=metadata.to_dict())

        return [doc]

    def _cleanse_html_text(self, html: str) -> str:
        """Cleanse HTML content by applying include/exclude selectors.

        Args:
            html (str): Raw HTML text.

        Returns:
            str: Cleansed text.
        """
        from bs4 import BeautifulSoup, Tag

        # Remove query strings from image URLs to avoid duplication
        html = self._strip_asset_cache_busters(html)
        soup = BeautifulSoup(html, "html.parser")

        # Drop unwanted tags
        for tag_name in self._cfg.strip_tags:
            for t in soup.find_all(tag_name):
                t.decompose()

        for selector in self._cfg.exclude_selectors:
            for t in soup.select(selector):
                t.decompose()

        # Include only selected tags
        include_selectors = self._cfg.include_selectors
        if include_selectors:
            included_nodes: list = []
            for selector in include_selectors:
                included_nodes.extend(soup.select(selector))

            seen = set()
            unique_nodes = []
            for node in included_nodes:
                key = id(node)

                if key in seen:
                    continue

                seen.add(key)
                unique_nodes.append(node)

            if unique_nodes:
                # Move only the "main content candidates" to a new soup
                new_soup = BeautifulSoup("<html><body></body></html>", "html.parser")
                body: Tag | list = new_soup.body or []
                for node in unique_nodes:
                    # Extract from the original soup and move to new_soup
                    body.append(node.extract())

                soup = new_soup

        # Remove excessive blank lines
        cleansed = [ln.strip() for ln in str(soup).splitlines()]
        cleansed = [ln for ln in cleansed if ln]

        return "\n".join(cleansed)

    def _strip_asset_cache_busters(self, html: str) -> str:
        """Remove cache busters from asset URLs in HTML.

        Args:
            html (str): Raw HTML text.

        Returns:
            str: HTML text with cache busters removed.
        """
        import re

        from ....core.exts import Exts

        exts = sorted(
            {
                ext.lstrip(".")
                for ext in Exts.IMAGE | {Exts.SVG} | Exts.AUDIO | Exts.VIDEO
            }
        )
        if not exts:
            return html

        # png|jpe?g|webp etc.
        ext_pattern = "|".join(
            ext.replace("+", r"\+").replace(".", r"\.") for ext in exts
        )
        pattern = rf"(\.(?:{ext_pattern}))\?[^\s\"'<>]+"

        return re.sub(pattern, r"\1", html)
