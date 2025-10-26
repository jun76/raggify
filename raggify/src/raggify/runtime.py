from __future__ import annotations

import threading
from typing import Optional

from .config.config_manager import ConfigManager
from .embed.embed import create_embed_manager
from .embed.embed_manager import EmbedManager
from .ingest.loader.file_loader import FileLoader
from .ingest.loader.html_loader import HTMLLoader
from .meta_store.meta_store import create_meta_store
from .meta_store.structured.structured import Structured
from .rerank.rerank import create_rerank_manager
from .rerank.rerank_manager import RerankManager
from .vector_store.vector_store import create_vector_store_manager
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

        self._embed_manager = None
        self._meta_store = None
        self._vector_store = None
        self._rerank_manager = None
        self._file_loader = None
        self._html_loader = None

    @property
    def cfg(self) -> ConfigManager:
        if self._cfg is None:
            self._cfg = ConfigManager()

        return self._cfg

    @property
    def embed_manager(self) -> EmbedManager:
        """EmbedManager を取得する。

        Returns:
            EmbedManager: 初期化済みの EmbedManager
        """
        if self._embed_manager is None:
            self._embed_manager = create_embed_manager(self.cfg)

        return self._embed_manager

    @property
    def meta_store(self) -> Structured:
        """Structured メタストアを取得する。

        Returns:
            Structured: 初期化済みメタストア
        """
        if self._meta_store is None:
            self._meta_store = create_meta_store(self.cfg)

        return self._meta_store

    @property
    def vector_store(self) -> VectorStoreManager:
        """VectorStoreManager を取得する。

        Returns:
            VectorStoreManager: 初期化済みベクトルストアマネージャ
        """
        if self._vector_store is None:
            self._vector_store = create_vector_store_manager(
                cfg=self.cfg,
                embed=self.embed_manager,
                meta_store=self.meta_store,
            )

        return self._vector_store

    @property
    def rerank_manager(self) -> RerankManager:
        """RerankManager を取得する。

        Returns:
            RerankManager: 初期化済みリランクマネージャ
        """
        if self._rerank_manager is None:
            self._rerank_manager = create_rerank_manager(self.cfg)

        return self._rerank_manager

    @property
    def file_loader(self) -> FileLoader:
        """FileLoader を取得する。

        Returns:
            FileLoader: 初期化済みファイルローダー
        """
        if self._file_loader is None:
            self._file_loader = FileLoader(
                chunk_size=self.cfg.ingest.chunk_size,
                chunk_overlap=self.cfg.ingest.chunk_overlap,
                store=self.vector_store,
            )

        return self._file_loader

    @property
    def html_loader(self) -> HTMLLoader:
        """HTMLLoader を取得する。

        Returns:
            HTMLLoader: 初期化済み HTML ローダー
        """
        if self._html_loader is None:
            self._html_loader = HTMLLoader(
                chunk_size=self.cfg.ingest.chunk_size,
                chunk_overlap=self.cfg.ingest.chunk_overlap,
                file_loader=self.file_loader,
                store=self.vector_store,
                user_agent=self.cfg.ingest.user_agent,
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
