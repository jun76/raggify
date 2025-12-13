from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from ....config.ingest_config import IngestConfig
from ....core.exts import Exts
from ....logger import logger
from ...parser import BaseParser
from ..util import arequest_get

if TYPE_CHECKING:
    from llama_index.core.schema import Document

__all__ = ["BaseWebPageReader"]


class BaseWebPageReader(ABC):
    """Reader abstract base for web pages that generates documents with parser."""

    def __init__(
        self, cfg: IngestConfig, asset_url_cache: set[str], parser: BaseParser
    ) -> None:
        """Constructor.

        Args:
            cfg (IngestConfig): Ingest configuration.
            asset_url_cache (set[str]): Cache of already processed asset URLs.
            parser (Parser): Parser instance.
        """
        self._cfg = cfg
        self._asset_url_cache = asset_url_cache
        self._parser = parser

    @abstractmethod
    async def aload_data(self, url: str) -> list[Document]:
        """Load data from a URL.

        Args:
            url (str): Target URL.

        Returns:
            list[Document]: List of documents read from the URL.
        """
        ...

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

    async def _adownload_direct_linked_file(
        self,
        url: str,
        allowed_exts: set[str],
        max_asset_bytes: int,
    ) -> Optional[str]:
        """Download a direct-linked file and return the local temp file path.

        Args:
            url (str): Target URL.
            allowed_exts (set[str]): Allowed extensions (lowercase with dot).
            max_asset_bytes (int): Max size in bytes.

        Returns:
            Optional[str]: Local temporary file path.
        """
        ext = Exts.get_ext(url)
        if ext not in allowed_exts:
            logger.warning(
                f"unsupported ext {ext}: {' '.join(allowed_exts)} are allowed."
            )
            return None

        try:
            res = await arequest_get(
                url=url,
                user_agent=self._cfg.user_agent,
                timeout_sec=self._cfg.timeout_sec,
                req_per_sec=self._cfg.req_per_sec,
            )
        except Exception as e:
            logger.exception(e)
            return None

        content_type = (res.headers.get("Content-Type") or "").lower()
        if "text/html" in content_type:
            logger.warning(f"skip asset (unexpected content-type): {content_type}")
            return None

        body = res.content or b""
        if len(body) > int(max_asset_bytes):
            logger.warning(
                f"skip asset (too large): {len(body)} Bytes > {int(max_asset_bytes)}"
            )
            return None

        # FIXME: issue #5 Handling MIME Types When Asset URL Extensions and
        # Actual Entities Mismatch in HTMLReader._adownload_direct_linked_file
        from ....core.utils import get_temp_path

        ext = Exts.get_ext(url)
        path = str(get_temp_path(seed=url, suffix=ext))
        try:
            with open(path, "wb") as f:
                f.write(body)
        except OSError as e:
            logger.warning(f"failed to save asset to temp file: {e}")
            return None

        return path

    def register_asset_url(self, url: str) -> bool:
        """Register an asset URL in the cache if it is new.

        Args:
            url (str): Asset URL.

        Returns:
            bool: True if added this time.
        """
        if url in self._asset_url_cache:
            return False

        self._asset_url_cache.add(url)

        return True

    async def aload_direct_linked_file(
        self,
        url: str,
        base_url: Optional[str] = None,
        max_asset_bytes: int = 100 * 1024 * 1024,
    ) -> list[Document]:
        """Create a document from a direct-linked file.

        Args:
            url (str): Target URL.
            base_url (Optional[str], optional): Base source URL. Defaults to None.
            max_asset_bytes (int, optional): Max size in bytes. Defaults to 100*1024*1024.

        Returns:
            list[Document]: Generated documents.
        """
        from ....core.metadata import MetaKeys as MK

        temp = await self._adownload_direct_linked_file(
            url=url,
            allowed_exts=self._parser.ingest_target_exts,
            max_asset_bytes=max_asset_bytes,
        )
        if temp is None:
            return []

        docs = await self._parser.aparse(temp)
        logger.debug(f"parsed {len(docs)} docs from downloaded asset: {url}")

        for doc in docs:
            meta = doc.metadata
            meta[MK.URL] = url
            meta[MK.BASE_SOURCE] = base_url or ""
            meta[MK.TEMP_FILE_PATH] = temp  # For cleanup

        return docs

    async def aload_direct_linked_files(
        self,
        urls: list[str],
        base_url: Optional[str] = None,
        max_asset_bytes: int = 100 * 1024 * 1024,
    ) -> list[Document]:
        """Create documents from multiple direct-linked files.

        Args:
            urls (list[str]): Target URLs.
            base_url (Optional[str], optional): Base source URL. Defaults to None.
            max_asset_bytes (int, optional): Max size in bytes. Defaults to 100*1024*1024.

        Returns:
            list[Document]: Generated documents.
        """
        docs = []
        for asset_url in urls:
            if not self.register_asset_url(asset_url):
                # Skip fetching identical assets
                continue

            asset_docs = await self.aload_direct_linked_file(
                url=asset_url,
                base_url=base_url,
                max_asset_bytes=max_asset_bytes,
            )
            if not asset_docs:
                logger.warning(f"failed to fetch from {asset_url}, skipped")
                continue

            docs.extend(asset_docs)

        return docs

    async def aload_html_text(self, url: str) -> tuple[list[Document], str]:
        """Generate documents from texts of an HTML page.

        Args:
            url (str): Target URL.

        Returns:
            tuple[list[Document], str]: Generated documents and the raw HTML.
        """
        from ....core.exts import Exts
        from ....core.metadata import MetaKeys as MK
        from ....core.utils import get_temp_path
        from ..util import afetch_text

        # Prefetch to avoid ingesting Not Found pages
        html = await afetch_text(
            url=url,
            user_agent=self._cfg.user_agent,
            timeout_sec=self._cfg.timeout_sec,
            req_per_sec=self._cfg.req_per_sec,
        )
        if not html:
            logger.warning(f"failed to fetch html from {url}, skipped")
            return [], ""

        html = self._cleanse_html_text(html)
        path = str(get_temp_path(seed=url, suffix=Exts.HTML))
        try:
            with open(path, "w") as f:
                f.write(html)
        except OSError as e:
            logger.warning(f"failed to save html to temp file: {e}")
            return [], ""

        docs = await self._parser.aparse(path)
        logger.debug(f"parsed {len(docs)} docs from html page: {url}")

        for doc in docs:
            doc.metadata[MK.URL] = url

        return docs, html
