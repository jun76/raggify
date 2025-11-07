from __future__ import annotations

import atexit
import threading
from typing import TYPE_CHECKING, Optional

from .config.config_manager import ConfigManager

if TYPE_CHECKING:
    from .document_store.document_store_manager import DocumentStoreManager
    from .embed.embed_manager import EmbedManager
    from .ingest.loader.file_loader import FileLoader
    from .ingest.loader.html_loader import HTMLLoader
    from .ingest_cache.ingest_cache_manager import IngestCacheStoreManager
    from .rerank.rerank_manager import RerankManager
    from .vector_store.vector_store_manager import VectorStoreManager


__all__ = ["get_runtime"]


_runtime: Runtime | None = None
_lock = threading.Lock()


class Runtime:
    """実行時プロセスのコンテキストで各種インスタンスを維持管理するためのクラス。"""

    def __init__(self) -> None:
        """コンストラクタ"""
        self._cfg: Optional[ConfigManager] = None
        self._embed_manager: Optional[EmbedManager] = None
        self._vector_store: Optional[VectorStoreManager] = None
        self._document_store: Optional[DocumentStoreManager] = None
        self._ingest_cache: Optional[IngestCacheStoreManager] = None
        self._rerank_manager: Optional[RerankManager] = None
        self._file_loader: Optional[FileLoader] = None
        self._html_loader: Optional[HTMLLoader] = None

    def build(self) -> None:
        self._release()
        self.touch()

    def rebuild(self) -> None:
        # メモリ上の設定値を生かす
        self._release(False)
        self.touch()

    def _release(self, with_cfg: bool = True) -> None:
        """既存のリソースを破棄する。

        Args:
            with_cfg (bool, optional): メモリ上の設定値も破棄するか。Defaults to True.
        """
        if with_cfg:
            self._cfg = None

        self._embed_manager = None
        self._vector_store = None
        self._document_store = None
        self._ingest_cache = None
        self._rerank_manager = None
        self._file_loader = None
        self._html_loader = None

    def touch(self) -> None:
        """各シングルトンの生成が未だであれば生成する。"""
        self.embed_manager
        self.vector_store
        self.document_store
        self.ingest_cache
        self.rerank_manager
        self.file_loader
        self.html_loader

    # 以下、シングルトンのインスタンス取得用
    @property
    def cfg(self) -> ConfigManager:
        if self._cfg is None:
            self._cfg = ConfigManager()

        return self._cfg

    @property
    def embed_manager(self) -> EmbedManager:
        if self._embed_manager is None:
            from .embed.embed import create_embed_manager

            self._embed_manager = create_embed_manager(self.cfg)

        return self._embed_manager

    @property
    def vector_store(self) -> VectorStoreManager:
        if self._vector_store is None:
            from .vector_store.vector_store import create_vector_store_manager

            self._vector_store = create_vector_store_manager(
                cfg=self.cfg, embed=self.embed_manager, docstore=self.document_store
            )

        return self._vector_store

    @property
    def document_store(self) -> DocumentStoreManager:
        if self._document_store is None:
            from .document_store.document_store import create_document_store_manager

            self._document_store = create_document_store_manager(self.cfg)

        return self._document_store

    @property
    def ingest_cache(self) -> IngestCacheStoreManager:
        if self._ingest_cache is None:
            from .ingest_cache.ingest_cache import create_ingest_cache_manager

            self._ingest_cache = create_ingest_cache_manager(
                cfg=self.cfg, embed=self.embed_manager
            )

        return self._ingest_cache

    @property
    def rerank_manager(self) -> RerankManager:
        if self._rerank_manager is None:
            from .rerank.rerank import create_rerank_manager

            self._rerank_manager = create_rerank_manager(self.cfg)

        return self._rerank_manager

    @property
    def file_loader(self) -> FileLoader:
        if self._file_loader is None:
            from .ingest.loader.file_loader import FileLoader

            self._file_loader = FileLoader(
                document_store=self.document_store,
                persist_dir=self.cfg.ingest.pipe_persist_dir,
            )

        return self._file_loader

    @property
    def html_loader(self) -> HTMLLoader:
        if self._html_loader is None:
            from .ingest.loader.html_loader import HTMLLoader

            self._html_loader = HTMLLoader(
                document_store=self.document_store,
                file_loader=self.file_loader,
                persist_dir=self.cfg.ingest.pipe_persist_dir,
                cfg=self.cfg.ingest,
            )

        return self._html_loader


def get_runtime() -> Runtime:
    """ランタイムシングルトンの getter。

    Returns:
        Runtime: ランタイム
    """
    global _runtime

    if _runtime is None:
        with _lock:
            if _runtime is None:
                _runtime = Runtime()

    return _runtime


def _shutdown_runtime() -> None:
    """ランタイムの終了処理"""
    global _runtime

    if _runtime is not None:
        try:
            _runtime._release()
        finally:
            _runtime = None


atexit.register(_shutdown_runtime)
