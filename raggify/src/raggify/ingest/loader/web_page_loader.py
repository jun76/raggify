from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from ...config.ingest_config import IngestConfig
from ...core.exts import Exts
from ...logger import logger
from .base_loader import BaseLoader

if TYPE_CHECKING:
    from llama_index.core.schema import Document, ImageNode, TextNode

    from ...llama_like.core.schema import AudioNode, VideoNode
    from ..parser import BaseParser

__all__ = ["WebPageLoader"]


class WebPageLoader(BaseLoader):
    """Loader for web pages that generates nodes."""

    def __init__(
        self,
        cfg: IngestConfig,
        parser: BaseParser,
        is_known_source: Optional[Callable[[str], bool]] = None,
    ):
        """Constructor.

        Args:
            cfg (IngestConfig): Ingest configuration.
            parser (Parser): Parser instance.
            is_known_source (Optional[Callable[[str], bool]]):
                Function to check if a source is known to skip. Defaults to None.
        """
        self._cfg = cfg
        self._parser = parser
        self._is_known_source = is_known_source

        # Do not include base_url in doc_id so identical URLs are treated
        # as the same document. Cache processed URLs in the same ingest run
        # so repeated assets are skipped without invoking pipeline.arun.
        self._asset_url_cache: set[str] = set()

        self.xml_schema_sitemap = "http://www.sitemaps.org/schemas/sitemap/0.9"

    def _parse_sitemap(self, raw_sitemap: str) -> list[str]:
        """Ported from SitemapReader in llama-index

        Args:
            raw_sitemap (str): Raw sitemap XML.

        Returns:
            list: List of URLs in the sitemap.
        """
        from xml.etree.ElementTree import fromstring

        sitemap = fromstring(raw_sitemap)
        sitemap_urls: list[str] = []

        for url in sitemap.findall(f"{{{self.xml_schema_sitemap}}}url"):
            location = url.find(f"{{{self.xml_schema_sitemap}}}loc").text  # type: ignore
            if location:
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
                user_agent=self._cfg.user_agent,
                timeout_sec=self._cfg.timeout_sec,
                req_per_sec=self._cfg.req_per_sec,
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
        from .web_page_reader.wikipedia_reader import MultiWikipediaReader

        reader = MultiWikipediaReader(
            cfg=self._cfg,
            asset_url_cache=self._asset_url_cache,
            parser=self._parser,
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
        from .web_page_reader.default_web_page_reader import DefaultWebPageReader

        reader = DefaultWebPageReader(
            cfg=self._cfg,
            asset_url_cache=self._asset_url_cache,
            parser=self._parser,
        )

        return await reader.aload_data(url)

    def _remove_query_params(self, uri: str) -> str:
        """Normalize URL query parameters by dropping configured keys.

        Args:
            uri (str): File path or URL string.

        Returns:
            str: URI with selected query keys removed.
        """
        from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

        strip_keys = {key.lower() for key in self._cfg.strip_query_keys}
        if not strip_keys:
            return uri

        parsed = urlparse(uri)
        if not parsed.query:
            return uri

        params = parse_qsl(parsed.query, keep_blank_values=True)
        filtered = [
            (key, value) for key, value in params if key.lower() not in strip_keys
        ]
        if len(filtered) == len(params):
            return uri

        return urlunparse(parsed._replace(query=urlencode(filtered)))

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

        if urlparse(url).scheme not in {"http", "https"}:
            logger.error("invalid URL. expected http(s)://*")
            return [], [], [], []

        if self._is_known_source is not None and self._is_known_source(url):
            logger.debug(f"skip already ingested URL: {url}")
            return [], [], [], []

        if not inloop:
            self._asset_url_cache.clear()

        url = self._remove_query_params(url)

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
