from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional
from urllib.parse import urljoin, urlparse

import requests

from ...config.ingest_config import IngestConfig
from ...core.exts import Exts
from ...logger import logger
from .loader import Loader

if TYPE_CHECKING:
    from llama_index.core.schema import Document, ImageNode, TextNode

    from ...llama.core.schema import AudioNode, VideoNode
    from .file_loader import FileLoader


class HTMLLoader(Loader):
    def __init__(
        self,
        file_loader: FileLoader,
        persist_dir: Optional[Path],
        cfg: IngestConfig,
    ):
        """Loader for HTML that generates nodes.

        Args:
            file_loader (FileLoader): Helper for file loading.
            persist_dir (Optional[Path]): Persist directory.
            cfg (IngestConfig): Ingest configuration.
        """
        super().__init__(persist_dir)

        self._file_loader = file_loader
        self._load_asset = cfg.load_asset
        self._req_per_sec = cfg.req_per_sec
        self._timeout_sec = cfg.timeout_sec
        self._user_agent = cfg.user_agent
        self._same_origin = cfg.same_origin

        # Do not include base_url in doc_id so identical URLs are treated
        # as the same document. Cache processed URLs in the same ingest run
        # so repeated assets are skipped without invoking pipeline.arun.
        self._asset_url_cache: set[str] = set()

    async def _arequest_get(self, url: str) -> requests.Response:
        """Async wrapper for HTTP GET.

        Args:
            url (str): Target URL.

        Raises:
            requests.HTTPError: On HTTP errors.
            RuntimeError: When fetching fails.

        Returns:
            requests.Response: Response object.
        """
        headers = {"User-Agent": self._user_agent}
        res: Optional[requests.Response] = None

        try:
            res = await asyncio.to_thread(
                requests.get,
                url,
                timeout=self._timeout_sec,
                headers=headers,
            )
            res.raise_for_status()
        except requests.HTTPError as e:
            status = res.status_code if res is not None else "unknown"
            raise requests.HTTPError(f"HTTP {status}: {str(e)}") from e
        except requests.RequestException as e:
            raise RuntimeError("failed to fetch url") from e
        finally:
            await asyncio.sleep(1 / self._req_per_sec)

        return res

    async def _afetch_text(
        self,
        url: str,
    ) -> str:
        """Fetch HTML and return the response text.

        Args:
            url (str): Target URL.

        Returns:
            str: Response body.
        """
        try:
            res = await self._arequest_get(url)
        except Exception as e:
            logger.exception(e)
            return ""

        return res.text

    def _sanitize_html_text(self, html: str) -> str:
        """Remove extra elements such as cache busters.

        Args:
            html (str): Raw HTML text.

        Returns:
            str: Sanitized text.
        """
        import re

        return re.sub(r"(\.(?:svg|png|jpe?g|webp))\?[^\s\"'<>]+", r"\1", html)

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
                if self._same_origin and (pu.scheme, pu.netloc) != (
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

    async def _adownload_direct_linked_file(
        self,
        url: str,
        allowed_exts: set[str],
        max_asset_bytes: int = 100 * 1024 * 1024,
    ) -> Optional[str]:
        """Download a direct-linked file and return the local temp file path.

        Args:
            url (str): Target URL.
            allowed_exts (set[str]): Allowed extensions (lowercase with dot).
            max_asset_bytes (int, optional): Max size in bytes. Defaults to 100*1024*1024.

        Returns:
            Optional[str]: Local temporary file path.
        """
        from ...core.metadata import get_temp_file_path_from

        if not Exts.endswith_exts(url, allowed_exts):
            logger.warning(f"unsupported ext. {' '.join(allowed_exts)} are allowed.")
            return None

        try:
            res = await self._arequest_get(url)
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

        ext = Exts.get_ext(url)
        path = get_temp_file_path_from(source=url, suffix=ext)
        try:
            with open(path, "wb") as f:
                f.write(body)
        except OSError as e:
            logger.exception(e)
            return None

        return path

    async def _aload_direct_linked_file(
        self, url: str, base_url: Optional[str] = None
    ) -> Optional[Document]:
        """Create a document from a direct-linked file.

        Args:
            url (str): Target URL.
            base_url (Optional[str], optional): Base source URL. Defaults to None.

        Returns:
            Optional[Document]: Generated document.
        """
        from llama_index.core.schema import Document

        from ...core.metadata import BasicMetaData

        temp_file_path = await self._adownload_direct_linked_file(
            url=url, allowed_exts=Exts.FETCH_TARGET
        )

        if temp_file_path is None:
            return None

        meta = BasicMetaData()
        meta.file_path = temp_file_path  # For MultiModalVectorStoreIndex
        meta.url = url
        meta.base_source = base_url or ""
        meta.temp_file_path = temp_file_path  # For cleanup

        return Document(text=url, metadata=meta.to_dict())

    def _register_asset_url(self, url: str) -> bool:
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

    async def _aload_html_asset_files(
        self,
        base_url: str,
        html: Optional[str] = None,
    ) -> list[Document]:
        """Load HTML and create documents from asset files.

        Args:
            base_url (str): Target URL.
            html (str): Prefetched HTML.

        Returns:
            list[Document]: Generated documents.
        """
        if html is None:
            html = await self._afetch_text(base_url)

        urls = self._gather_asset_links(
            html=html, base_url=base_url, allowed_exts=Exts.FETCH_TARGET
        )

        docs = []
        for url in urls:
            if not self._register_asset_url(url):
                # Skip fetching identical assets
                continue

            doc = await self._aload_direct_linked_file(url=url, base_url=base_url)
            if doc is None:
                logger.warning(f"failed to fetch from {url}, skipped")
                continue

            docs.append(doc)

        return docs

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
        import html2text
        from llama_index.core.schema import Document

        from ...core.metadata import MetaKeys as MK

        if urlparse(url).scheme not in {"http", "https"}:
            logger.error("invalid URL. expected http(s)://*")
            return []

        docs = []
        if Exts.endswith_exts(url, Exts.FETCH_TARGET):
            # Direct-linked file
            if self._register_asset_url(url):
                doc = await self._aload_direct_linked_file(url)
                if doc is None:
                    logger.warning(f"failed to fetch from {url}, skipped")
                else:
                    docs.append(doc)
        else:
            # Prefetch to avoid ingesting Not Found pages
            html = await self._afetch_text(url)
            if not html:
                logger.warning(f"failed to fetch html from {url}, skipped")
                return []

            # Body text
            text = self._sanitize_html_text(html)
            text = html2text.html2text(text)
            doc = Document(text=text, metadata={MK.URL: url})
            docs.append(doc)

            if self._load_asset:
                # Asset files
                docs.extend(await self._aload_html_asset_files(base_url=url, html=html))

        logger.debug(f"loaded {len(docs)} docs from {url}")

        return docs

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
            return await self._asplit_docs_modality(docs=docs, is_canceled=is_canceled)

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
                break

            temp = await self._aload_from_site(url)
            docs.extend(temp)

        return await self._asplit_docs_modality(docs=docs, is_canceled=is_canceled)

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
                break
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
