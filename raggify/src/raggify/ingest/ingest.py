from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Optional, Sequence

from ..logger import logger
from ..runtime import get_runtime as rt

if TYPE_CHECKING:
    from ..vector_store.vector_store_manager import VectorStoreManager
    from .loader.file_loader import FileLoader
    from .loader.html_loader import HTMLLoader

__all__ = [
    "ingest_path",
    "aingest_path",
    "ingest_path_list",
    "aingest_path_list",
    "ingest_url",
    "aingest_url",
    "ingest_url_list",
    "aingest_url_list",
]


def _read_list(path: str) -> list[str]:
    """path や URL のリストをファイルから読み込む

    Args:
        path (str): リストのパス

    Returns:
        list[str]: 読み込んだリスト
    """
    lst = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lst = f.readlines()
    except OSError as e:
        logger.warning(f"failed to read config file: {e}")

    return lst


def ingest_path(
    path: str,
    store: Optional[VectorStoreManager] = None,
    file_loader: Optional[FileLoader] = None,
) -> None:
    """ローカルパス（ディレクトリ、ファイル）からコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        path (str): 対象パス
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        file_loader (Optional[FileLoader]): ファイル読み込み用。Defaults to None.
    """
    asyncio.run(aingest_path(path=path, store=store, file_loader=file_loader))


async def aingest_path(
    path: str,
    store: Optional[VectorStoreManager] = None,
    file_loader: Optional[FileLoader] = None,
) -> None:
    """ローカルパス（ディレクトリ、ファイル）から非同期でコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        path (str): 対象パス
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        file_loader (Optional[FileLoader]): ファイル読み込み用。Defaults to None.
    """
    store = store or rt().vector_store
    file_loader = file_loader or rt().file_loader

    nodes = await file_loader.aload_from_path(path)
    await store.aupsert_nodes(nodes)


def ingest_path_list(
    lst: str | Sequence[str],
    store: Optional[VectorStoreManager] = None,
    file_loader: Optional[FileLoader] = None,
) -> None:
    """パスリスト内の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        lst (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        file_loader (Optional[FileLoader]): ファイル読み込み用。Defaults to None.
    """
    asyncio.run(
        aingest_path_list(
            lst=lst,
            store=store,
            file_loader=file_loader,
        )
    )


async def aingest_path_list(
    lst: str | Sequence[str],
    store: Optional[VectorStoreManager] = None,
    file_loader: Optional[FileLoader] = None,
) -> None:
    """パスリスト内の複数パスから非同期でコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        file_loader (Optional[FileLoader]): ファイル読み込み用。Defaults to None.
    """
    if isinstance(lst, str):
        lst = _read_list(lst)

    store = store or rt().vector_store
    file_loader = file_loader or rt().file_loader

    nodes = await file_loader.aload_from_paths(list(lst))
    await store.aupsert_nodes(nodes)


def ingest_url(
    url: str,
    store: Optional[VectorStoreManager] = None,
    html_loader: Optional[HTMLLoader] = None,
) -> None:
    """URL からコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        url (str): 対象 URL
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        html_loader (Optional[HTMLLoader]): HTML 読み込み用。Defaults to None.
    """
    asyncio.run(aingest_url(url=url, store=store, html_loader=html_loader))


async def aingest_url(
    url: str,
    store: Optional[VectorStoreManager] = None,
    html_loader: Optional[HTMLLoader] = None,
) -> None:
    """URL から非同期でコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        url (str): 対象 URL
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        html_loader (Optional[HTMLLoader]): HTML 読み込み用。Defaults to None.
    """
    store = store or rt().vector_store
    html_loader = html_loader or rt().html_loader

    nodes = await html_loader.aload_from_url(url)
    await store.aupsert_nodes(nodes)


def ingest_url_list(
    lst: str | Sequence[str],
    store: Optional[VectorStoreManager] = None,
    html_loader: Optional[HTMLLoader] = None,
) -> None:
    """URL リスト内の複数サイトからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        lst (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        html_loader (Optional[HTMLLoader]): HTML 読み込み用。Defaults to None.
    """
    asyncio.run(
        aingest_url_list(
            lst=lst,
            store=store,
            html_loader=html_loader,
        )
    )


async def aingest_url_list(
    lst: str | Sequence[str],
    store: Optional[VectorStoreManager] = None,
    html_loader: Optional[HTMLLoader] = None,
) -> None:
    """URL リスト内の複数サイトから非同期でコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        lst (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
        store (Optional[VectorStoreManager]): ベクトルストア。Defaults to None.
        html_loader (Optional[HTMLLoader]): HTML 読み込み用。Defaults to None.
    """
    if isinstance(lst, str):
        lst = _read_list(lst)

    store = store or rt().vector_store
    html_loader = html_loader or rt().html_loader

    nodes = await html_loader.aload_from_urls(list(lst))
    await store.aupsert_nodes(nodes)
