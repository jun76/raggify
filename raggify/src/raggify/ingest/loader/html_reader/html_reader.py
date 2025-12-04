from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from urllib.parse import urljoin, urlparse

from ....config.general_config import GeneralConfig
from ....config.ingest_config import IngestConfig
from ....core.exts import Exts
from ....logger import logger
from ..util import arequest_get

if TYPE_CHECKING:
    from llama_index.core.schema import Document


class HTMLReader:
    def __init__(
        self,
        icfg: IngestConfig,
        gcfg: GeneralConfig,
        asset_url_cache: set[str],
        ingest_target_exts: set[str],
    ) -> None:
        """Loader for HTML that generates nodes.

        Args:
            icfg (IngestConfig): Ingest configuration.
            gcfg (GeneralConfig): General configuration.
            asset_url_cache (set[str]): Cache of already processed asset URLs.
            ingest_target_exts (set[str]): Allowed extensions for ingestion.
        """
        self._icfg = icfg
        self._gcfg = gcfg
        self._asset_url_cache = asset_url_cache
        self._ingest_target_exts = ingest_target_exts

    def cleanse_html_content(self, html: str) -> str:
        """Cleanse HTML content by applying include/exclude selectors.

        Args:
            html (str): Raw HTML text.

        Returns:
            str: Sanitized text.
        """
        import re

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Drop unwanted tags
        for tag_name in self._icfg.strip_tags:
            for t in soup.find_all(tag_name):
                t.decompose()

        for selector in self._icfg.exclude_selectors:
            for t in soup.select(selector):
                t.decompose()

        # Include only selected tags
        include_selectors = self._icfg.include_selectors
        if include_selectors:
            included_nodes: list = []
            for selector in include_selectors:
                included_nodes.extend(soup.select(selector))

            seen = set()
            unique_nodes = []
            for node in included_nodes:
                key = id(node)

                if key in seen:
                    continue

                seen.add(key)
                unique_nodes.append(node)

            if unique_nodes:
                # Move only the "main content candidates" to a new soup
                new_soup = BeautifulSoup("<html><body></body></html>", "html.parser")
                body = new_soup.body or []
                for node in unique_nodes:
                    # Extract from the original soup and move to new_soup
                    body.append(node.extract())

                soup = new_soup

        # Remove excessive blank lines
        cleansed = [ln.strip() for ln in str(soup).splitlines()]
        cleansed = [ln for ln in cleansed if ln]
        cleansed = "\n".join(cleansed)

        return re.sub(r"(\.(?:svg|png|jpe?g|webp))\?[^\s\"'<>]+", r"\1", cleansed)

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

    def gather_asset_links(
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
                if self._icfg.same_origin and (pu.scheme, pu.netloc) != (
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
        from ....core.utils import get_temp_file_path_from

        ext = Exts.get_ext(url)
        if ext not in allowed_exts:
            logger.warning(
                f"unsupported ext {ext}: {' '.join(allowed_exts)} are allowed."
            )
            return None

        try:
            res = await arequest_get(
                url=url,
                user_agent=self._icfg.user_agent,
                timeout_sec=self._icfg.timeout_sec,
                req_per_sec=self._icfg.req_per_sec,
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
        ext = Exts.get_ext(url)
        path = get_temp_file_path_from(source=url, suffix=ext)
        try:
            with open(path, "wb") as f:
                f.write(body)
        except OSError as e:
            logger.exception(e)
            return None

        return path

    async def aload_direct_linked_file(
        self,
        url: str,
        base_url: Optional[str] = None,
        max_asset_bytes: int = 100 * 1024 * 1024,
    ) -> list[Document]:
        """Create a document from a direct-linked file.

        Args:
            url (str): Target URL.
            base_url (Optional[str], optional): Base source URL. Defaults to None.
            max_asset_bytes (int, optional): Max size in bytes. Defaults to 100*1024*1024.

        Returns:
            list[Document]: Generated documents.
        """
        from ....core.metadata import BasicMetaData
        from ..parser import DefaultParser

        temp = await self._adownload_direct_linked_file(
            url=url,
            allowed_exts=self._ingest_target_exts,
            max_asset_bytes=max_asset_bytes,
        )
        if temp is None:
            return []

        parser = DefaultParser(self._gcfg, self._ingest_target_exts)
        docs = await parser.aparse(temp)

        parsed_docs = []
        for doc in docs:
            meta = BasicMetaData().from_dict(doc.metadata)
            meta.url = url
            meta.base_source = base_url or ""
            meta.temp_file_path = temp  # For cleanup
            doc.metadata = meta.to_dict()
            parsed_docs.append(doc)

        return parsed_docs
