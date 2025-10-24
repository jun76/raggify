from __future__ import annotations

import threading
from typing import Optional

from .config import cfg
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

__all__ = [
    "reload",
    "get_embed_manager",
    "get_meta_store",
    "get_vector_store",
    "get_rerank_manager",
    "get_file_loader",
    "get_html_loader",
]

_init_lock = threading.RLock()

_embed_manager: Optional[EmbedManager] = None
_meta_store: Optional[Structured] = None
_vector_store: Optional[VectorStoreManager] = None
_rerank_manager: Optional[RerankManager] = None
_file_loader: Optional[FileLoader] = None
_html_loader: Optional[HTMLLoader] = None


def reload(reload_config: bool = True) -> None:
    """既存のリソースを破棄し、必要に応じて設定を再読み込みする。

    Args:
        reload_config (bool, optional): True の場合、ConfigManager もリロードする。Defaults to True.
    """
    global _embed_manager, _meta_store, _vector_store, _rerank_manager
    global _file_loader, _html_loader

    with _init_lock:
        if reload_config:
            cfg.reload()

        _embed_manager = None
        _meta_store = None
        _vector_store = None
        _rerank_manager = None
        _file_loader = None
        _html_loader = None


def get_embed_manager() -> EmbedManager:
    """EmbedManager を取得する。

    Returns:
        EmbedManager: 初期化済みの EmbedManager
    """
    global _embed_manager

    with _init_lock:
        if _embed_manager is None:
            _embed_manager = create_embed_manager()

    return _embed_manager


def get_meta_store() -> Structured:
    """Structured メタストアを取得する。

    Returns:
        Structured: 初期化済みメタストア
    """
    global _meta_store

    with _init_lock:
        if _meta_store is None:
            _meta_store = create_meta_store()

    return _meta_store


def get_vector_store() -> VectorStoreManager:
    """VectorStoreManager を取得する。

    Returns:
        VectorStoreManager: 初期化済みベクトルストアマネージャ
    """
    global _vector_store

    with _init_lock:
        if _vector_store is None:
            _vector_store = create_vector_store_manager(
                embed=get_embed_manager(),
                meta_store=get_meta_store(),
            )

    return _vector_store


def get_rerank_manager() -> RerankManager:
    """RerankManager を取得する。

    Returns:
        RerankManager: 初期化済みリランクマネージャ
    """
    global _rerank_manager

    with _init_lock:
        if _rerank_manager is None:
            _rerank_manager = create_rerank_manager()

    return _rerank_manager


def get_file_loader() -> FileLoader:
    """FileLoader を取得する。

    Returns:
        FileLoader: 初期化済みファイルローダー
    """
    global _file_loader

    with _init_lock:
        if _file_loader is None:
            _file_loader = FileLoader(
                chunk_size=cfg.ingest.chunk_size,
                chunk_overlap=cfg.ingest.chunk_overlap,
                store=get_vector_store(),
            )

    return _file_loader


def get_html_loader() -> HTMLLoader:
    """HTMLLoader を取得する。

    Returns:
        HTMLLoader: 初期化済み HTML ローダー
    """
    global _html_loader

    with _init_lock:
        if _html_loader is None:
            _html_loader = HTMLLoader(
                chunk_size=cfg.ingest.chunk_size,
                chunk_overlap=cfg.ingest.chunk_overlap,
                file_loader=get_file_loader(),
                store=get_vector_store(),
                user_agent=cfg.ingest.user_agent,
            )

    return _html_loader
