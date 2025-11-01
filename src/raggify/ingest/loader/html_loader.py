from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional
from urllib.parse import urljoin, urlparse

import requests

from ...config.default_settings import DefaultSettings as DS
from ...core.exts import Exts
from ...logger import logger
from .loader import Loader

if TYPE_CHECKING:
    from llama_index.core.schema import Document, ImageNode, TextNode

    from ...document_store.document_store_manager import DocumentStoreManager
    from ...llama.core.schema import AudioNode
    from .file_loader import FileLoader


class HTMLLoader(Loader):
    def __init__(
        self,
        document_store: DocumentStoreManager,
        file_loader: FileLoader,
        load_asset: bool = DS.LOAD_ASSET,
        req_per_sec: int = DS.REQ_PER_SEC,
        timeout_sec: int = DS.TIMEOUT_SEC,
        user_agent: str = DS.USER_AGENT,
        same_origin: bool = DS.SAME_ORIGIN,
    ):
        """HTML を読み込み、ノードを生成するためのクラス。

        Args:
            document_store (DocumentStoreManager): ドキュメントストア管理
            file_loader (FileLoader): ファイル読み込み用
            load_asset (bool, optional): アセットを読み込むか。Defaults to DS.LOAD_ASSET.
            req_per_sec (int, optional): 秒間リクエスト数。Defaults to DS.REQ_PER_SEC.
            timeout_sec (int, optional): タイムアウト秒。Defaults to DS.TIMEOUT_SEC.
            user_agent (str, optional): GET リクエスト時の user agent。Defaults to DS.USER_AGENT.
            same_origin (bool, optional): 同一オリジンのみ対象とするか。Defaults to DS.SAME_ORIGIN.
        """
        super().__init__(document_store)
        self._file_loader = file_loader
        self._load_asset = load_asset
        self._req_per_sec = req_per_sec
        self._timeout_sec = timeout_sec
        self._user_agent = user_agent
        self._same_origin = same_origin

    async def _arequest_get(self, url: str) -> requests.Response:
        """HTTP GET を実行する非同期ラッパー。

        Args:
            url (str): 対象 URL

        Raises:
            requests.HTTPError: GET 時の例外
            RuntimeError: フェッチ失敗

        Returns:
            requests.Response: 取得した Response データ
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
        """HTML を取得し、テキストを返す。

        Args:
            url (str): 取得先 URL

        Returns:
            str: レスポンス本文
        """
        try:
            res = await self._arequest_get(url)
        except Exception as e:
            logger.exception(e)
            return ""

        return res.text

    def _gather_asset_links(
        self,
        html: str,
        base_url: str,
        allowed_exts: set[str],
        limit: int = 20,
    ) -> list[str]:
        """HTML からアセット URL を収集する。

        Args:
            html (str): HTML 文字列
            base_url (str): 相対 URL 解決用の基準 URL
            allowed_exts (set[str]): 許可される拡張子集合（ドット付き小文字）
            limit (int, optional): 返却する最大件数.Defaults to 20.

        Returns:
            list[str]: 収集した絶対 URL
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
        """直リンクのファイルをダウンロードし、ローカルの一時ファイルパスを返す。

        Args:
            url (str): 対象 URL
            allowed_exts (set[str]): 許可される拡張子集合（ドット付き小文字）
            max_asset_bytes (int, optional): データサイズ上限。Defaults to 100*1024*1024.

        Returns:
            Optional[str]: ローカルの一時ファイルパス
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
        """直リンクのファイルからドキュメントを生成する。

        Args:
            url (str): 対象 URL
            base_url (Optional[str], optional): source の取得元を指定する場合。Defaults to None.

        Returns:
            Optional[Document]: 生成したドキュメント
        """
        from ...core.metadata import BasicMetaData

        temp_file_path = await self._adownload_direct_linked_file(
            url=url, allowed_exts=Exts.FETCH_TARGET
        )

        if temp_file_path is None:
            return None

        meta = BasicMetaData()
        meta.file_path = temp_file_path  # MultiModalVectorStoreIndex 参照用
        meta.url = url
        meta.base_source = base_url or ""
        meta.temp_file_path = temp_file_path  # 削除用

        return Document(text=url, metadata=meta.to_dict())

    async def _aload_html_asset_files(
        self,
        base_url: str,
        html: Optional[str] = None,
    ) -> list[Document]:
        """HTML を読み込み、アセットファイルからドキュメントを生成する。

        Args:
            base_url (str): 対象 URL
            html (str): プリフェッチした html

        Returns:
            list[Document]: 生成したドキュメント
        """
        if html is None:
            html = await self._afetch_text(base_url)

        urls = self._gather_asset_links(
            html=html, base_url=base_url, allowed_exts=Exts.FETCH_TARGET
        )

        if not urls:
            logger.info("no asset files found")
            return []

        logger.info(f"{len(urls)} asset files found")

        # セマフォで同時実行数を制限
        semaphore = asyncio.Semaphore(self._req_per_sec)

        async def fetch_with_limit(url: str) -> Optional[Document]:
            async with semaphore:
                return await self._aload_direct_linked_file(url=url, base_url=base_url)

        tasks = [fetch_with_limit(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs = []
        for url, doc in zip(urls, results):
            if isinstance(doc, Exception) or doc is None:
                logger.warning(f"failed to fetch from {url}, skipped")
                continue

            docs.append(doc)

        return docs

    async def _aload_from_site(
        self,
        url: str,
    ) -> list[Document]:
        """単一サイトからコンテンツを取得し、ドキュメントを生成する。

        Args:
            url (str): 対象 URL

        Returns:
            list[Document]: 生成したドキュメント
        """
        import html2text

        from ...core.metadata import MetaKeys as MK

        if urlparse(url).scheme not in {"http", "https"}:
            logger.error("invalid URL. expected http(s)://*")
            return []

        docs = []
        if Exts.endswith_exts(url, Exts.FETCH_TARGET):
            # 直リンクファイル
            doc = await self._aload_direct_linked_file(url)
            if doc is None:
                logger.warning(f"failed to fetch from {url}, skipped")
            else:
                docs.append(doc)
        else:
            # Not Found ページを ingest しないように下見
            html = await self._afetch_text(url)
            if not html:
                logger.warning(f"failed to fetch html from {url}, skipped")
                return []

            # 本文テキスト
            text = html2text.html2text(html)
            doc = Document(text=text, metadata={MK.URL: url})
            docs.append(doc)

            if self._load_asset:
                # アセットファイル
                docs.extend(await self._aload_html_asset_files(base_url=url, html=html))

        logger.debug(f"loaded {len(docs)} docs from {url}")

        return docs

    async def aload_from_url(
        self,
        url: str,
    ) -> tuple[list[TextNode], list[ImageNode], list[AudioNode]]:
        """URL からコンテンツを取得し、ノードを生成する。
        サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

        Args:
            url (str): 対象 URL

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode]]:
                テキストノード、画像ノード、音声ノード
        """
        from llama_index.readers.web.sitemap.base import SitemapReader

        # サイトマップ以外は単一のサイトとして読み込み
        if not Exts.endswith_exts(url, Exts.SITEMAP):
            docs = await self._aload_from_site(url)
            return await self._asplit_docs_modality(docs)

        # 以下、サイトマップの解析と読み込み
        try:
            loader = SitemapReader()
            urls = loader._parse_sitemap(url)
        except Exception as e:
            logger.exception(e)
            return [], [], []

        docs = []
        for url in urls:
            temp = await self._aload_from_site(url)
            docs.extend(temp)

        return await self._asplit_docs_modality(docs)

    async def aload_from_urls(
        self,
        urls: list[str],
    ) -> tuple[list[TextNode], list[ImageNode], list[AudioNode]]:
        """URL リスト内の複数サイトからコンテンツを取得し、ノードを生成する。

        Args:
            urls (list[str]): URL リスト

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode]]:
                テキストノード、画像ノード、音声ノード
        """
        texts = []
        images = []
        audios = []
        for url in urls:
            try:
                temp_text, temp_image, temp_audio = await self.aload_from_url(url)
                texts.extend(temp_text)
                images.extend(temp_image)
                audios.extend(temp_audio)
            except Exception as e:
                logger.exception(e)
                continue

        return texts, images, audios
