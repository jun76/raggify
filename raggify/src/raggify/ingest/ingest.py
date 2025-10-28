from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ..core.event import async_loop_runner
from ..logger import logger

if TYPE_CHECKING:
    from ..runtime import Runtime

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


def _rt() -> Runtime:
    """遅延ロード用ゲッター。

    Returns:
        Runtime: ランタイム
    """
    from ..runtime import get_runtime

    return get_runtime()


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


def ingest_path(path: str) -> None:
    """ローカルパス（ディレクトリ、ファイル）からコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        path (str): 対象パス
    """
    async_loop_runner.run(lambda: aingest_path(path))


async def aingest_path(path: str) -> None:
    """ローカルパス（ディレクトリ、ファイル）から非同期でコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        path (str): 対象パス
    """
    store = _rt().vector_store
    file_loader = _rt().file_loader

    nodes = await file_loader.aload_from_path(path)
    await store.aupsert_nodes(nodes)


def ingest_path_list(lst: str | Sequence[str]) -> None:
    """パスリスト内の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        lst (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
    """
    async_loop_runner.run(lambda: aingest_path_list(lst))


async def aingest_path_list(lst: str | Sequence[str]) -> None:
    """パスリスト内の複数パスから非同期でコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
    """
    if isinstance(lst, str):
        lst = _read_list(lst)

    store = _rt().vector_store
    file_loader = _rt().file_loader

    nodes = await file_loader.aload_from_paths(list(lst))
    await store.aupsert_nodes(nodes)


def ingest_url(url: str) -> None:
    """URL からコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        url (str): 対象 URL
    """
    async_loop_runner.run(lambda: aingest_url(url))


async def aingest_url(url: str) -> None:
    """URL から非同期でコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        url (str): 対象 URL
    """
    store = _rt().vector_store
    html_loader = _rt().html_loader

    nodes = await html_loader.aload_from_url(url)
    await store.aupsert_nodes(nodes)


def ingest_url_list(lst: str | Sequence[str]) -> None:
    """URL リスト内の複数サイトからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        lst (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
    """
    async_loop_runner.run(lambda: aingest_url_list(lst))


async def aingest_url_list(lst: str | Sequence[str]) -> None:
    """URL リスト内の複数サイトから非同期でコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        lst (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
    """
    if isinstance(lst, str):
        lst = _read_list(lst)

    store = _rt().vector_store
    html_loader = _rt().html_loader

    nodes = await html_loader.aload_from_urls(list(lst))
    await store.aupsert_nodes(nodes)
