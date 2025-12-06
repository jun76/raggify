from __future__ import annotations

from urllib.parse import urljoin, urlparse

from llama_index.core.schema import Document

from ....config.ingest_config import IngestConfig
from ....logger import logger
from ...parser import BaseParser
from .base_web_page_reader import BaseWebPageReader

__all__ = ["DefaultWebPageReader"]


class DefaultWebPageReader(BaseWebPageReader):
    """Reader for web pages that generates documents."""

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
        super().__init__(
            cfg=cfg,
            asset_url_cache=asset_url_cache,
            parser=parser,
        )

    async def aload_data(self, url: str) -> list[Document]:
        """Load data from a URL.

        Args:
            url (str): Target URL.

        Returns:
            list[Document]: List of documents read from the URL.
        """
        from ....core.exts import Exts

        # Direct linked file
        if Exts.endswith_exts(url, self._parser.ingest_target_exts):
            if not self.register_asset_url(url):
                return []

            docs = await self.aload_direct_linked_file(
                url=url, base_url=url, max_asset_bytes=self._cfg.max_asset_bytes
            )
            if docs is None:
                logger.warning(f"failed to fetch from {url}")
                return []

            return docs

        text_docs, html = await self._aload_texts(url)
        logger.debug(f"loaded {len(text_docs)} text docs from {url}")

        asset_docs = (
            await self._aload_assets(url=url, html=html) if self._cfg.load_asset else []
        )
        logger.debug(f"loaded {len(asset_docs)} asset docs from {url}")

        return text_docs + asset_docs

    async def _aload_texts(self, url: str) -> tuple[list[Document], str]:
        """Generate documents from texts of a web page.

        Args:
            url (str): Target URL.

        Returns:
            tuple[list[Document], str]: Generated documents and the raw HTML.
        """
        return await self.aload_html_text(url)

    async def _aload_assets(self, url: str, html: str) -> list[Document]:
        """Generate documents from assets of a web page.

        Args:
            url (str): Target URL.
            html (str): Raw HTML content.

        Returns:
            list[Document]: Generated documents.
        """
        urls = self._gather_asset_links(
            html=html, base_url=url, allowed_exts=self._parser.ingest_target_exts
        )

        return await self.aload_direct_linked_files(
            urls=urls,
            base_url=url,
            max_asset_bytes=self._cfg.max_asset_bytes,
        )

    def _gather_asset_links(
        self,
        html: str,
        base_url: str,
        allowed_exts: set[str],
        limit: int = 20,
    ) -> list[str]:
        """Collect asset URLs from HTML.

        Args:
            html (str): HTML string.
            base_url (str): Base URL for resolving relatives.
            allowed_exts (set[str]): Allowed extensions (lowercase with dot).
            limit (int, optional): Max results. Defaults to 20.

        Returns:
            list[str]: Absolute URLs collected.
        """
        from bs4 import BeautifulSoup

        from ....core.exts import Exts

        seen = set()
        out = []
        base = urlparse(base_url)

        def add(u: str) -> None:
            if not u:
                return

            try:
                absu = urljoin(base_url, u)
                if absu in seen:
                    return

                pu = urlparse(absu)
                if self._cfg.same_origin and (pu.scheme, pu.netloc) != (
                    base.scheme,
                    base.netloc,
                ):
                    return

                path = pu.path.lower()
                if Exts.endswith_exts(path, allowed_exts):
                    seen.add(absu)
                    out.append(absu)
            except Exception:
                return

        soup = BeautifulSoup(html, "html.parser")

        for img in soup.find_all("img"):
            add(img.get("src"))  # type: ignore

        for a in soup.find_all("a"):
            add(a.get("href"))  # type: ignore

        for src in soup.find_all("source"):
            ss = src.get("srcset")  # type: ignore
            if ss:
                cand = ss.split(",")[0].strip().split(" ")[0]  # type: ignore
                add(cand)

        return out[: max(0, limit)]
