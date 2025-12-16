from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

from ....config.ingest_config import IngestConfig
from ....core.exts import Exts
from ....logger import logger
from ...parser import BaseParser

if TYPE_CHECKING:
    from llama_index.core.schema import Document

__all__ = ["BaseWebPageReader"]


class BaseWebPageReader(ABC):
    """Reader abstract base for web pages that generates documents with parser."""

    def __init__(
        self,
        cfg: IngestConfig,
        asset_url_cache: set[str],
        parser: BaseParser,
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
        from ..util import arequest_get

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
        max_asset_bytes: int,
        base_url: Optional[str] = None,
    ) -> list[Document]:
        """Create a document from a direct-linked file.

        Args:
            url (str): Target URL.
            max_asset_bytes (int): Max size in bytes.
            base_url (Optional[str], optional): Base source URL. Defaults to None.

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
        max_asset_bytes: int,
        base_url: Optional[str] = None,
    ) -> list[Document]:
        """Create documents from multiple direct-linked files.

        Args:
            urls (list[str]): Target URLs.
            max_asset_bytes (int): Max size in bytes.
            base_url (Optional[str], optional): Base source URL. Defaults to None.

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
                max_asset_bytes=max_asset_bytes,
                base_url=base_url,
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
