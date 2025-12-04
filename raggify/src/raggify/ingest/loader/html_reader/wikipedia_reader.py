from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from llama_index.core.schema import Document

from ....logger import logger
from .html_reader import HTMLReader

if TYPE_CHECKING:
    from wikipedia import WikipediaPage

    from ....config.general_config import GeneralConfig
    from ....config.ingest_config import IngestConfig


class MultiWikipediaReader(HTMLReader):
    """Multimodal Wikipedia reader."""

    def __init__(
        self,
        icfg: IngestConfig,
        gcfg: GeneralConfig,
        asset_url_cache: set[str],
        ingest_target_exts: set[str],
    ) -> None:
        """Initialize with parameters.
        Args:
            icfg (IngestConfig): Ingest configuration.
            gcfg (GeneralConfig): General configuration.
            asset_url_cache (set[str]): Cache of already processed asset URLs.
            ingest_target_exts (set[str]): Allowed extensions for ingestion.
        """
        super().__init__(
            icfg=icfg,
            gcfg=gcfg,
            asset_url_cache=asset_url_cache,
            ingest_target_exts=ingest_target_exts,
        )
        self._load_asset = icfg.load_asset

    async def aload_data(self, url: str, **load_kwargs: Any) -> List[Document]:
        """
        Load data from Wikipedia.

        Args:
            url (str): Wikipedia page URL.
            **load_kwargs: Additional arguments for wikipedia.page().

        Returns:
            List[Document]: List of documents read from Wikipedia.
        """
        wiki_page = self._fetch_wiki_page(url, **load_kwargs)

        text_docs = await self._aload_texts(wiki_page)
        logger.debug(f"loaded {len(text_docs)} text docs from {wiki_page.url}")

        asset_docs = await self._aload_assets(wiki_page) if self._load_asset else []
        logger.debug(f"loaded {len(asset_docs)} asset docs from {wiki_page.url}")

        return text_docs + asset_docs

    def _fetch_wiki_page(self, url: str, **load_kwargs: Any) -> WikipediaPage:
        """Fetch a Wikipedia page based on the URL and additional loading arguments.

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

        return wikipedia.page(page, **load_kwargs)

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
        docs = []
        for url in page.images:
            if not self.register_asset_url(url):
                # Skip fetching identical assets
                continue

            doc = await self.aload_direct_linked_file(url=url, base_url=page.url)
            if doc is None:
                logger.warning(f"failed to fetch from {url}, skipped")
                continue

            docs.append(doc)

        return docs
