from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import cfg

if TYPE_CHECKING:
    from .structured.structured import Structured

__all__ = ["create_meta_store"]


def create_meta_store() -> Structured:
    """メタデータ用ストアのインスタンスを生成する。

    Raises:
        RuntimeError: インスタンス生成に失敗

    Returns:
        Structured: メタデータ用ストア
    """
    from .structured.sqlite_structured import SQLiteStructured

    try:
        meta_store = SQLiteStructured(cfg.meta_store.meta_store_path)
    except Exception as e:
        raise RuntimeError(f"failed to prepare metadata store: {e}") from e

    return meta_store
