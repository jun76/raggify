from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional
from urllib.parse import urlparse

from ...config.ingest_config import IngestConfig
from ...core.exts import Exts
from ...logger import logger
from .loader import Loader

if TYPE_CHECKING:
    from llama_index.core.schema import Document, ImageNode, TextNode

    from ...llama.core.schema import AudioNode, VideoNode


class HTMLLoader(Loader):
    def __init__(
        self,
        persist_dir: Optional[Path],
        cfg: IngestConfig,
    ):
        """Loader for HTML that generates nodes.

        Args:
            persist_dir (Optional[Path]): Persist directory.
            cfg (IngestConfig): Ingest configuration.
        """
        super().__init__(persist_dir)

        self._cfg = cfg

        # Do not include base_url in doc_id so identical URLs are treated
        # as the same document. Cache processed URLs in the same ingest run
        # so repeated assets are skipped without invoking pipeline.arun.
        self._asset_url_cache: set[str] = set()

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
        if urlparse(url).scheme not in {"http", "https"}:
            logger.error("invalid URL. expected http(s)://*")
            return []

        if self._is_wikipedia(url):
            docs = await self._aload_from_site_wikipedia(url)
        else:
            docs = await self._aload_from_site_default(url)

        logger.debug(f"loaded {len(docs)} docs from {url}")

        return docs

    def _is_wikipedia(self, url: str) -> bool:
        """Check if the URL is a Wikipedia site.

        Args:
            url (str): Target URL.

        Returns:
            bool: True if the URL is a Wikipedia site.
        """
        return "wikipedia.org" in url

    async def _aload_from_site_wikipedia(
        self,
        url: str,
    ) -> list[Document]:
        """Fetch content from a single site and create documents.

        Args:
            url (str): Target URL.

        Returns:
            list[Document]: Generated documents.
        """
        from .html_reader.wikipedia_reader import MultiWikipediaReader

        reader = MultiWikipediaReader(
            cfg=self._cfg, asset_url_cache=self._asset_url_cache
        )

        return await reader.aload_data(url)

    async def _aload_from_site_default(
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

        reader = DefaultHTMLReader(cfg=self._cfg, asset_url_cache=self._asset_url_cache)

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
        from llama_index.readers.web.sitemap.base import SitemapReader

        if not inloop:
            self._asset_url_cache.clear()

        # For non-sitemaps, treat as a single site
        if not Exts.endswith_exts(url, Exts.SITEMAP):
            docs = await self._aload_from_site(url)
            return await self._asplit_docs_modality(docs)

        # Parse and ingest sitemap
        try:
            loader = SitemapReader()
            raw_sitemap = loader._load_sitemap(url)
            urls = loader._parse_sitemap(raw_sitemap)
        except Exception as e:
            logger.exception(e)
            return [], [], [], []

        docs = []
        for url in urls:
            if is_canceled():
                logger.info("Job is canceled, aborting batch processing")
                return [], [], [], []

            temp = await self._aload_from_site(url)
            docs.extend(temp)

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
