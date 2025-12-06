from __future__ import annotations

from typing import TYPE_CHECKING

from llama_index.core.schema import Document

from ....ingest.parser import BaseParser
from ....logger import logger
from .base_web_page_reader import BaseWebPageReader

if TYPE_CHECKING:
    from wikipedia import WikipediaPage

    from ....config.ingest_config import IngestConfig

__all__ = ["MultiWikipediaReader"]


class MultiWikipediaReader(BaseWebPageReader):
    """Reader for Wikipedia that generates documents."""

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
            parser (BaseParser): Parser instance.
        """
        super().__init__(cfg=cfg, asset_url_cache=asset_url_cache, parser=parser)
        self._load_asset = cfg.load_asset

    async def aload_data(self, url: str) -> list[Document]:
        """Load data from Wikipedia.

        Args:
            url (str): Wikipedia page URL.

        Returns:
            list[Document]: list of documents read from Wikipedia.
        """
        wiki_page = self._fetch_wiki_page(url)

        text_docs = await self._aload_texts(wiki_page)
        logger.debug(f"loaded {len(text_docs)} text docs from {wiki_page.url}")

        asset_docs = await self._aload_assets(wiki_page) if self._load_asset else []
        logger.debug(f"loaded {len(asset_docs)} asset docs from {wiki_page.url}")

        return text_docs + asset_docs

    def _fetch_wiki_page(self, url: str) -> WikipediaPage:
        """Fetch a Wikipedia page based on the URL.

        Args:
            url (str): Wikipedia page URL.

        Raises:
            ValueError: If the language prefix is not supported.

        Returns:
            WikipediaPage: Wikipedia page object.
        """
        import wikipedia

        lang_prefix = url.split(".wikipedia.org")[0].split("//")[-1]
        if lang_prefix.lower() != "en":
            if lang_prefix.lower() in wikipedia.languages():
                wikipedia.set_lang(lang_prefix.lower())
            else:
                raise ValueError(
                    f"Language prefix '{lang_prefix}' for Wikipedia is not supported. "
                    "Check supported languages at https://en.wikipedia.org/wiki/List_of_Wikipedias."
                )

        page = url.split("/wiki/")[-1]

        return wikipedia.page(page)

    async def _aload_texts(self, page: WikipediaPage) -> list[Document]:
        """Generate documents from texts of a Wikipedia page.

        Args:
            page (WikipediaPage): Wikipedia page.

        Returns:
            list[Document]: Generated documents.
        """
        from ....core.metadata import MetaKeys as MK

        doc = Document(id_=page.pageid, text=page.content)
        doc.metadata[MK.URL] = page.url

        return [doc]

    async def _aload_assets(self, page: WikipediaPage) -> list[Document]:
        """Generate documents from assets of a Wikipedia page.

        Args:
            page (WikipediaPage): Wikipedia page.

        Returns:
            list[Document]: Generated documents.
        """
        return await self.aload_direct_linked_files(
            urls=page.images,
            base_url=page.url,
            max_asset_bytes=self._cfg.max_asset_bytes,
        )
