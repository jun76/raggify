from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ...config.general_config import GeneralConfig
from ...config.ingest_config import IngestConfig
from ...core.exts import Exts
from ...logger import logger
from .loader import Loader

if TYPE_CHECKING:
    from llama_index.core.schema import Document, ImageNode, TextNode

    from ...llama_like.core.schema import AudioNode, VideoNode


class HTMLLoader(Loader):
    def __init__(
        self,
        icfg: IngestConfig,
        gcfg: GeneralConfig,
        ingest_target_exts: set[str],
    ):
        """Loader for HTML that generates nodes.

        Args:
            icfg (IngestConfig): Ingest configuration.
            gcfg (GeneralConfig): General configuration.
            ingest_target_exts (set[str]): Allowed extensions for ingestion.
        """
        self._icfg = icfg
        self._gcfg = gcfg
        self._ingest_target_exts = ingest_target_exts

        # Do not include base_url in doc_id so identical URLs are treated
        # as the same document. Cache processed URLs in the same ingest run
        # so repeated assets are skipped without invoking pipeline.arun.
        self._asset_url_cache: set[str] = set()

        self.xml_schema_sitemap = "http://www.sitemaps.org/schemas/sitemap/0.9"

    def _parse_sitemap(self, raw_sitemap: str) -> list:
        """Ported from SitemapReader in llama-index

        Args:
            raw_sitemap (str): Raw sitemap XML.

        Returns:
            list: List of URLs in the sitemap.
        """
        from xml.etree.ElementTree import fromstring

        sitemap = fromstring(raw_sitemap)
        sitemap_urls = []

        for url in sitemap.findall(f"{{{self.xml_schema_sitemap}}}url"):
            location = url.find(f"{{{self.xml_schema_sitemap}}}loc").text  # type: ignore
            sitemap_urls.append(location)

        return sitemap_urls

    async def _aload_from_sitemap(
        self,
        url: str,
        is_canceled: Callable[[], bool],
    ) -> list[Document]:
        """Fetch content from a sitemap and create documents.

        Args:
            url (str): Target URL.
            is_canceled (Callable[[], bool]): Whether this job has been canceled.

        Returns:
            list[Document]: Generated documents.
        """
        from .util import afetch_text

        try:
            raw_sitemap = await afetch_text(
                url=url,
                user_agent=self._icfg.user_agent,
                timeout_sec=self._icfg.timeout_sec,
                req_per_sec=self._icfg.req_per_sec,
            )
            urls = self._parse_sitemap(raw_sitemap)
        except Exception as e:
            logger.exception(e)
            return []

        docs = []
        for url in urls:
            if is_canceled():
                logger.info("Job is canceled, aborting batch processing")
                return []

            temp = await self._aload_from_site(url)
            docs.extend(temp)

        return docs

    async def _aload_from_wikipedia(
        self,
        url: str,
    ) -> list[Document]:
        """Fetch content from a Wikipedia site and create documents.

        Args:
            url (str): Target URL.

        Returns:
            list[Document]: Generated documents.
        """
        from .html_reader.wikipedia_reader import MultiWikipediaReader

        reader = MultiWikipediaReader(
            icfg=self._icfg,
            gcfg=self._gcfg,
            asset_url_cache=self._asset_url_cache,
            ingest_target_exts=self._ingest_target_exts,
        )

        return await reader.aload_data(url)

    async def _aload_from_site(
        self,
        url: str,
    ) -> list[Document]:
        """Fetch content from a single site and create documents.

        Args:
            url (str): Target URL.

        Returns:
            list[Document]: Generated documents.
        """
        from .html_reader.default_html_reader import DefaultHTMLReader

        reader = DefaultHTMLReader(
            icfg=self._icfg,
            gcfg=self._gcfg,
            asset_url_cache=self._asset_url_cache,
            ingest_target_exts=self._ingest_target_exts,
        )

        return await reader.aload_data(url)

    async def aload_from_url(
        self,
        url: str,
        is_canceled: Callable[[], bool],
        inloop: bool = False,
    ) -> tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
        """Fetch content from a URL and generate nodes.

        For sitemaps (.xml), traverse the tree to ingest multiple sites.

        Args:
            url (str): Target URL.
            is_canceled (Callable[[], bool]): Whether this job has been canceled.
            inloop (bool, optional): Whether called inside an upper URL loop. Defaults to False.

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
                Text, image, audio, and video nodes.
        """
        from urllib.parse import urlparse

        if not inloop:
            self._asset_url_cache.clear()

        if urlparse(url).scheme not in {"http", "https"}:
            logger.error("invalid URL. expected http(s)://*")
            return [], [], [], []

        if Exts.endswith_exts(url, Exts.SITEMAP):
            docs = await self._aload_from_sitemap(url=url, is_canceled=is_canceled)
        elif "wikipedia.org" in url:
            docs = await self._aload_from_wikipedia(url)
        else:
            docs = await self._aload_from_site(url)

        logger.debug(f"loaded {len(docs)} docs from {url}")

        return await self._asplit_docs_modality(docs)

    async def aload_from_urls(
        self,
        urls: list[str],
        is_canceled: Callable[[], bool],
    ) -> tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
        """Fetch content from multiple URLs and generate nodes.

        Args:
            urls (list[str]): URL list.
            is_canceled (Callable[[], bool]): Whether this job has been canceled.

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
                Text, image, audio, and video nodes.
        """
        self._asset_url_cache.clear()

        texts = []
        images = []
        audios = []
        videos = []
        for url in urls:
            if is_canceled():
                logger.info("Job is canceled, aborting batch processing")
                return [], [], [], []
            try:
                temp_text, temp_image, temp_audio, temp_video = (
                    await self.aload_from_url(
                        url=url, is_canceled=is_canceled, inloop=True
                    )
                )
                texts.extend(temp_text)
                images.extend(temp_image)
                audios.extend(temp_audio)
                videos.extend(temp_video)
            except Exception as e:
                logger.exception(e)
                continue

        return texts, images, audios, videos
