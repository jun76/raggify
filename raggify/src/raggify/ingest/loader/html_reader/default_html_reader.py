from __future__ import annotations

from llama_index.core.schema import Document

from ....config.ingest_config import IngestConfig
from ....logger import logger
from .html_reader import HTMLReader


class DefaultHTMLReader(HTMLReader):
    def __init__(self, cfg: IngestConfig, asset_url_cache: set[str]) -> None:
        """Default HTML reader.

        Args:
            cfg (IngestConfig): Ingest configuration.
            asset_url_cache (set[str]): Cache of already processed asset URLs.
        """
        super().__init__(cfg, asset_url_cache)
        self._load_asset = cfg.load_asset

    async def aload_data(self, url: str) -> list[Document]:
        """Load data from a URL.

        Args:
            url (str): Target URL.

        Returns:
            list[Document]: List of documents read from the URL.
        """

        text_docs, html = await self._aload_texts(url)
        logger.debug(f"loaded {len(text_docs)} text docs from {url}")

        asset_docs = (
            await self._aload_assets(url=url, html=html) if self._load_asset else []
        )
        logger.debug(f"loaded {len(asset_docs)} asset docs from {url}")

        return text_docs + asset_docs

    async def _aload_texts(self, url: str) -> tuple[list[Document], str]:
        """Generate documents from texts of an HTML page.

        Args:
            url (str): Target URL.

        Returns:
            tuple[list[Document], str]: Generated documents and the raw HTML.
        """
        import html2text

        from ....core.metadata import MetaKeys as MK

        docs = []

        # Prefetch to avoid ingesting Not Found pages
        html = await self.afetch_text(url)
        if not html:
            logger.warning(f"failed to fetch html from {url}, skipped")
            return [], ""

        # Body text
        text = self.sanitize_html_text(html)
        text = html2text.html2text(text)
        doc = Document(text=text, metadata={MK.URL: url})
        docs.append(doc)

        return docs, html

    async def _aload_assets(self, url: str, html: str) -> list[Document]:
        """Generate documents from assets of an HTML page.

        Args:
            url (str): Target URL.
            html (str): Raw HTML content.

        Returns:
            list[Document]: Generated documents.
        """
        from ....core.exts import Exts

        if html is None:
            html = await self.afetch_text(url)

        urls = self.gather_asset_links(
            html=html, base_url=url, allowed_exts=Exts.FETCH_TARGET
        )

        docs = []
        for url in urls:
            if not self.register_asset_url(url):
                # Skip fetching identical assets
                continue

            doc = await self.aload_direct_linked_file(url=url, base_url=url)
            if doc is None:
                logger.warning(f"failed to fetch from {url}, skipped")
                continue

            docs.append(doc)

        return docs
