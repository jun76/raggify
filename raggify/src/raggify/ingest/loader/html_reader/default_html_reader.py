from __future__ import annotations

from llama_index.core.schema import Document

from ....config.ingest_config import IngestConfig
from ....logger import logger
from ...parser import BaseParser
from .html_reader import HTMLReader


class DefaultHTMLReader(HTMLReader):
    def __init__(
        self,
        cfg: IngestConfig,
        asset_url_cache: set[str],
        parser: BaseParser,
    ) -> None:
        """Default HTML reader.

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

        if Exts.endswith_exts(url, self._parser.ingest_target_exts):
            if not self.register_asset_url(url):
                return []

            # Direct linked file
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
        """Generate documents from texts of an HTML page.

        Args:
            url (str): Target URL.

        Returns:
            tuple[list[Document], str]: Generated documents and the raw HTML.
        """
        import html2text

        from ....core.metadata import MetaKeys as MK
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

        # Body text
        text = self.cleanse_html_content(html)
        text = html2text.html2text(text)
        doc = Document(text=text, metadata={MK.URL: url})

        return [doc], html

    async def _aload_assets(self, url: str, html: str) -> list[Document]:
        """Generate documents from assets of an HTML page.

        Args:
            url (str): Target URL.
            html (str): Raw HTML content.

        Returns:
            list[Document]: Generated documents.
        """
        from ..util import afetch_text

        if html is None:
            html = await afetch_text(
                url=url,
                user_agent=self._cfg.user_agent,
                timeout_sec=self._cfg.timeout_sec,
                req_per_sec=self._cfg.req_per_sec,
            )

        urls = self.gather_asset_links(
            html=html, base_url=url, allowed_exts=self._parser.ingest_target_exts
        )

        docs = []
        for asset_url in urls:
            if not self.register_asset_url(asset_url):
                # Skip fetching identical assets
                continue

            asset_docs = await self.aload_direct_linked_file(
                url=asset_url,
                base_url=asset_url,
                max_asset_bytes=self._cfg.max_asset_bytes,
            )
            if not asset_docs:
                logger.warning(f"failed to fetch from {asset_url}, skipped")
                continue

            docs.extend(asset_docs)

        return docs
