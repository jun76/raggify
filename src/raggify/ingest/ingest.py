from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from llama_index.core.ingestion import IngestionPipeline

from ..core.event import async_loop_runner
from ..embed.embed_manager import Modality
from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.ingestion.cache import IngestionCache
    from llama_index.core.schema import TransformComponent
    from llama_index.core.storage.docstore.types import BaseDocumentStore
    from llama_index.core.vector_stores.types import BasePydanticVectorStore

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


def _create_pipeline(
    transformations: list[TransformComponent],
    vector_store: Optional[BasePydanticVectorStore],
    docstore: Optional[BaseDocumentStore],
    cache: Optional[IngestionCache],
) -> IngestionPipeline:
    """モダリティ毎の IngestionPipeline を構築する。

    Args:
        transformations (list[TransformComponent]): 変換一式
        vector_store (Optional[BasePydanticVectorStore]): ベクトルストア
        docstore (Optional[BaseDocumentStore]): ドキュメントストア
        cache (Optional[IngestionCache]): ingestion キャッシュ

    Returns:
        IngestionPipeline: パイプライン
    """
    return IngestionPipeline(
        transformations=transformations,
        vector_store=vector_store,
        docstore=docstore,
        cache=cache,
    )


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
    from .trans import AddChunkIndexTransform

    vs = _rt().vector_store
    embed = _rt().embed_manager
    ds = _rt().document_store
    ics = _rt().ingest_cache_store
    file_loader = _rt().file_loader

    text_docs, image_docs, audio_docs = await file_loader.aload_from_path(path)

    text_pipeline = _create_pipeline(
        transformations=[
            AddChunkIndexTransform(),
            embed.get_container(Modality.TEXT).embed,
        ],
        vector_store=vs.get_container(Modality.TEXT).store,
        docstore=ds.get_container(Modality.TEXT).store,
        cache=ics.get_container(Modality.TEXT).store,
    )

    await text_pipeline.arun(documents=text_docs)
    await image_pipeline.arun(documents=image_docs)
    await audio_pipeline.arun(documents=audio_docs)


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
