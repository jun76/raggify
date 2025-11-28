from __future__ import annotations

from typing import TYPE_CHECKING, Optional
from urllib.parse import urljoin, urlparse

from ....config.ingest_config import IngestConfig
from ....core.exts import Exts
from ....logger import logger
from ..util import arequest_get

if TYPE_CHECKING:
    from llama_index.core.schema import Document


class HTMLReader:
    def __init__(
        self, cfg: IngestConfig, asset_url_cache: set[str], ingest_target_exts: set[str]
    ) -> None:
        """Loader for HTML that generates nodes.

        Args:
            cfg (IngestConfig): Ingest configuration.
            asset_url_cache (set[str]): Cache of already processed asset URLs.
            ingest_target_exts (set[str]): Allowed extensions for ingestion.
        """
        self._cfg = cfg
        self._asset_url_cache = asset_url_cache
        self._ingest_target_exts = ingest_target_exts

    def sanitize_html_text(self, html: str) -> str:
        """Remove extra elements such as cache busters.

        Args:
            html (str): Raw HTML text.

        Returns:
            str: Sanitized text.
        """
        import re

        return re.sub(r"(\.(?:svg|png|jpe?g|webp))\?[^\s\"'<>]+", r"\1", html)

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
                user_agent=self._cfg.user_agent,
                timeout_sec=self._cfg.timeout_sec,
                req_per_sec=self._cfg.req_per_sec,
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

        # FIXME:
        # Bedrock's MIME type check is strict, and if the MIME type you specified
        # is incorrect, it returns the following error:
        #
        #  An error occurred (ValidationException) when calling the InvokeModel
        #  operation: The detected file MIME type image/png does not match the
        #  expected type image/webp. Reformat your input and try again.
        #
        # Therefore, if the target URL is misleading (e.g., the URL specifies
        # .webp but the actual content is png), we must inspect the magic numbers
        # within the binary. However, since other providers do not require MIME
        # types in the first place and still succeed in embedding, we will treat
        # the URL as valid as before (even if the actual content differs) and
        # ignore this error.
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
    ) -> Optional[Document]:
        """Create a document from a direct-linked file.

        Args:
            url (str): Target URL.
            base_url (Optional[str], optional): Base source URL. Defaults to None.
            max_asset_bytes (int, optional): Max size in bytes. Defaults to 100*1024*1024.

        Returns:
            Optional[Document]: Generated document.
        """
        from llama_index.core.schema import Document

        from ....core.metadata import BasicMetaData

        temp = await self._adownload_direct_linked_file(
            url=url,
            allowed_exts=self._ingest_target_exts,
            max_asset_bytes=max_asset_bytes,
        )
        if temp is None:
            return None

        meta = BasicMetaData()
        meta.file_path = temp  # For MultiModalVectorStoreIndex
        meta.url = url
        meta.base_source = base_url or ""
        meta.temp_file_path = temp  # For cleanup

        return Document(text=url, metadata=meta.to_dict())
