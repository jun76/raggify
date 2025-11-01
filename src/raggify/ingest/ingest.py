from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from ..core.event import async_loop_runner
from ..embed.embed_manager import Modality
from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.schema import ImageNode, TextNode

    from ..llama.core.schema import AudioNode
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
            temp = []
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                temp.append(stripped)
            lst = temp
    except OSError as e:
        logger.warning(f"failed to read config file: {e}")

    return lst


async def _aupsert_nodes(
    text_nodes: list[TextNode],
    image_nodes: list[ImageNode],
    audio_nodes: list[AudioNode],
):
    """ノードをアップサートする。

    Args:
        text_nodes (list[TextNode]): テキストノード
        image_nodes (list[ImageNode]): 画像ノード
        audio_nodes (list[AudioNode]): 音声ノード
    """
    import asyncio

    from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
    from llama_index.core.node_parser import SentenceSplitter

    from .transform import (
        AddChunkIndexTransform,
        make_audio_embed_transform,
        make_image_embed_transform,
        make_text_embed_transform,
    )

    rt = _rt()
    embed = rt.embed_manager
    vs = rt.vector_store
    ics = rt.ingest_cache_store
    ds = rt.document_store

    # 後段パイプ。テキスト分割とモダリティ毎の埋め込み＋ストア格納。
    # キャッシュ管理も。
    tasks = []
    if text_nodes:
        text_pipe = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=rt.cfg.ingest.chunk_size,
                    chunk_overlap=rt.cfg.ingest.chunk_overlap,
                    include_metadata=True,
                ),
                AddChunkIndexTransform(),
                make_text_embed_transform(embed=embed),
            ],
            vector_store=vs.get_container(Modality.TEXT).store,
            cache=ics.get_container(Modality.TEXT).store,
            docstore=ds.store,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )
        tasks.append(text_pipe.arun(nodes=text_nodes))
        ds.store.persist()

    if image_nodes:
        image_pipe = IngestionPipeline(
            transformations=[
                make_image_embed_transform(embed=embed),
            ],
            vector_store=vs.get_container(Modality.IMAGE).store,
            cache=ics.get_container(Modality.IMAGE).store,
            docstore=ds.store,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )
        tasks.append(image_pipe.arun(nodes=image_nodes))
        ds.store.persist()

    if audio_nodes:
        audio_pipe = IngestionPipeline(
            transformations=[
                make_audio_embed_transform(embed=embed),
            ],
            vector_store=vs.get_container(Modality.AUDIO).store,
            cache=ics.get_container(Modality.AUDIO).store,
            docstore=ds.store,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )
        tasks.append(audio_pipe.arun(nodes=audio_nodes))
        ds.store.persist()

    if tasks:
        await asyncio.gather(*tasks)


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
    file_loader = _rt().file_loader

    text_nodes, image_nodes, audio_nodes = await file_loader.aload_from_path(path)
    await _aupsert_nodes(
        text_nodes=text_nodes, image_nodes=image_nodes, audio_nodes=audio_nodes
    )


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

    file_loader = _rt().file_loader

    text_nodes, image_nodes, audio_nodes = await file_loader.aload_from_paths(list(lst))
    await _aupsert_nodes(
        text_nodes=text_nodes, image_nodes=image_nodes, audio_nodes=audio_nodes
    )


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
    html_loader = _rt().html_loader

    text_nodes, image_nodes, audio_nodes = await html_loader.aload_from_url(url)
    await _aupsert_nodes(
        text_nodes=text_nodes, image_nodes=image_nodes, audio_nodes=audio_nodes
    )


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

    html_loader = _rt().html_loader

    text_nodes, image_nodes, audio_nodes = await html_loader.aload_from_urls(list(lst))
    await _aupsert_nodes(
        text_nodes=text_nodes, image_nodes=image_nodes, audio_nodes=audio_nodes
    )
