from __future__ import annotations

import atexit
import threading
from typing import TYPE_CHECKING, Optional

from .config.config_manager import ConfigManager

if TYPE_CHECKING:  # pragma: no cover
    from .embed.embed_manager import EmbedManager
    from .ingest.loader.file_loader import FileLoader
    from .ingest.loader.html_loader import HTMLLoader
    from .meta_store.structured.structured import Structured
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
        self._meta_store: Optional[Structured] = None
        self._vector_store: Optional[VectorStoreManager] = None
        self._rerank_manager: Optional[RerankManager] = None
        self._file_loader: Optional[FileLoader] = None
        self._html_loader: Optional[HTMLLoader] = None

    def reload(self, reload_config: bool = True) -> None:
        """既存のリソースを破棄し、必要に応じて設定を再読み込みする。

        Args:
            reload_config (bool, optional): ConfigManager もリロードするか。Defaults to True.
        """
        if reload_config:
            self.cfg.reload()

        self.close()

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
    def meta_store(self) -> Structured:
        if self._meta_store is None:
            from .meta_store.meta_store import create_meta_store

            self._meta_store = create_meta_store(self.cfg)

        return self._meta_store

    @property
    def vector_store(self) -> VectorStoreManager:
        if self._vector_store is None:
            from .vector_store.vector_store import create_vector_store_manager

            self._vector_store = create_vector_store_manager(
                cfg=self.cfg,
                embed=self.embed_manager,
                meta_store=self.meta_store,
            )

        return self._vector_store

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
                chunk_size=self.cfg.ingest.chunk_size,
                chunk_overlap=self.cfg.ingest.chunk_overlap,
                store=self.vector_store,
            )

        return self._file_loader

    @property
    def html_loader(self) -> HTMLLoader:
        if self._html_loader is None:
            from .ingest.loader.html_loader import HTMLLoader

            self._html_loader = HTMLLoader(
                chunk_size=self.cfg.ingest.chunk_size,
                chunk_overlap=self.cfg.ingest.chunk_overlap,
                file_loader=self.file_loader,
                store=self.vector_store,
                user_agent=self.cfg.ingest.user_agent,
            )

        return self._html_loader

    def close(self) -> None:
        """保持しているリソースを解放する。"""
        self._embed_manager = None
        self._vector_store = None
        self._rerank_manager = None
        self._file_loader = None
        self._html_loader = None

        if self._meta_store is not None:
            try:
                self._meta_store.close()
            finally:
                self._meta_store = None


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
            _runtime.close()
        finally:
            _runtime = None


atexit.register(_shutdown_runtime)
