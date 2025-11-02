from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from ..core.event import async_loop_runner
from ..embed.embed_manager import Modality
from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.schema import (
        BaseNode,
        ImageNode,
        TextNode,
        TransformComponent,
    )

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


async def _arun_docstore_pipe(
    nodes: list[BaseNode],
    transformations: list[TransformComponent] | None,
    modality: Modality,
) -> Optional[Sequence[BaseNode]]:
    """ドキュメント重複管理用パイプラインを実行する。

    Args:
        nodes (list[BaseNode]): ノード

    Returus:
        Optional[Sequence[BaseNode]]: 重複ドキュメントフィルター後のノード
    """
    import os

    from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline

    rt = _rt()
    vs = rt.vector_store
    ics = rt.ingest_cache_store
    ds = rt.document_store

    if not nodes:
        return None

    pipe = IngestionPipeline(
        transformations=transformations,
        vector_store=vs.get_container(modality).store,
        cache=ics.get_container(modality).store,
        docstore=ds.store,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )

    path = ics.persist_path
    if path and os.path.exists(path):
        pipe.load(path)

    filtered_nodes = await pipe.arun(nodes=nodes, store_doc_text=False)

    if path:
        pipe.persist(path)

    return filtered_nodes


async def _arun_text_docstore_pipe(
    nodes: list[TextNode],
) -> Optional[Sequence[BaseNode]]:
    """テキストノードのパイプラインを実行する。

    Args:
        nodes (list[TextNode]): テキストノード

    Returus:
        Optional[Sequence[BaseNode]]: 重複ドキュメントフィルター後のノード
    """
    from llama_index.core.node_parser import SentenceSplitter

    from .transform import AddChunkIndexTransform, make_text_embed_transform

    rt = _rt()
    return await _arun_docstore_pipe(
        nodes=[node for node in nodes],
        transformations=[
            SentenceSplitter(
                chunk_size=rt.cfg.ingest.chunk_size,
                chunk_overlap=rt.cfg.ingest.chunk_overlap,
                include_metadata=True,
            ),
            AddChunkIndexTransform(),
            make_text_embed_transform(rt.embed_manager),
        ],
        modality=Modality.TEXT,
    )


async def _arun_image_docstore_pipe(
    nodes: list[ImageNode],
) -> Optional[Sequence[BaseNode]]:
    """画像ノードのパイプラインを実行する。

    Args:
        nodes (list[ImageNode]): 画像ノード

    Returus:
        Optional[Sequence[BaseNode]]: 重複ドキュメントフィルター後のノード
    """
    from .transform import make_image_embed_transform

    rt = _rt()
    return await _arun_docstore_pipe(
        nodes=[node for node in nodes],
        transformations=[
            make_image_embed_transform(rt.embed_manager),
        ],
        modality=Modality.IMAGE,
    )


async def _arun_audio_docstore_pipe(
    nodes: list[AudioNode],
) -> Optional[Sequence[BaseNode]]:
    """音声ノードのパイプラインを実行する。

    Args:
        nodes (list[AudioNode]): 音声ノード

    Returus:
        Optional[Sequence[BaseNode]]: 重複ドキュメントフィルター後のノード
    """
    from .transform import make_audio_embed_transform

    rt = _rt()
    return await _arun_docstore_pipe(
        nodes=[node for node in nodes],
        transformations=[
            make_audio_embed_transform(rt.embed_manager),
        ],
        modality=Modality.AUDIO,
    )


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

    # 後段パイプ。テキスト分割とモダリティ毎の埋め込み＋ストア格納。
    # キャッシュ管理も。
    tasks = [
        _arun_text_docstore_pipe(text_nodes),
        _arun_image_docstore_pipe(image_nodes),
        _arun_audio_docstore_pipe(audio_nodes),
    ]

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
