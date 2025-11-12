from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from llama_index.core.async_utils import asyncio_run
from llama_index.core.ingestion import IngestionPipeline

from ..embed.embed_manager import Modality
from ..logger import logger
from ..runtime import get_runtime as _rt

if TYPE_CHECKING:
    from llama_index.core.schema import (
        BaseNode,
        ImageNode,
        TextNode,
        TransformComponent,
    )

    from ..llama.core.schema import AudioNode


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


def _build_text_pipeline(persist_dir: Optional[Path]) -> IngestionPipeline:
    """テキスト用パイプラインを構築する。

    Args:
        persist_dir (Optional[Path]): 永続化ディレクトリ

    Returns:
        IngestionPipeline: パイプライン
    """
    from llama_index.core.node_parser import SentenceSplitter

    from .transform import AddChunkIndexTransform, make_text_embed_transform

    rt = _rt()
    transformations: list[TransformComponent] = [
        SentenceSplitter(
            chunk_size=rt.cfg.ingest.chunk_size,
            chunk_overlap=rt.cfg.ingest.chunk_overlap,
            include_metadata=True,
        ),
        AddChunkIndexTransform(),
        make_text_embed_transform(rt.embed_manager),
    ]

    return rt.build_pipeline(
        transformations=transformations, modality=Modality.TEXT, persist_dir=persist_dir
    )


def _build_image_pipeline(persist_dir: Optional[Path]) -> IngestionPipeline:
    """画像用パイプラインを構築する。

    Args:
        persist_dir (Optional[Path]): 永続化ディレクトリ

    Returns:
        IngestionPipeline: パイプライン
    """
    from .transform import make_image_embed_transform

    rt = _rt()
    transformations: list[TransformComponent] = [
        make_image_embed_transform(rt.embed_manager),
    ]

    return rt.build_pipeline(
        transformations=transformations,
        modality=Modality.IMAGE,
        persist_dir=persist_dir,
    )


def _build_audio_pipeline(persist_dir: Optional[Path]) -> IngestionPipeline:
    """音声用パイプラインを構築する。

    Args:
        persist_dir (Optional[Path]): 永続化ディレクトリ

    Returns:
        IngestionPipeline: パイプライン
    """
    from .transform import make_audio_embed_transform

    rt = _rt()
    transformations: list[TransformComponent] = [
        make_audio_embed_transform(rt.embed_manager),
    ]

    return rt.build_pipeline(
        transformations=transformations,
        modality=Modality.AUDIO,
        persist_dir=persist_dir,
    )


async def _process_batches(
    nodes: Sequence[BaseNode],
    modality: Modality,
    persist_dir: Optional[Path],
    batch_size: int,
    is_canceled: Callable[[], bool],
) -> None:
    """大量ノードのアップサートで長時間ブロックしないようにバッチ化する。

    Args:
        nodes (Sequence[BaseNode]): ノード
        modality (Modality): モダリティ
        persist_dir (Optional[Path]): 永続化ディレクトリ
        batch_size (int): バッチサイズ
        is_canceled (Callable[[], bool]): このジョブがキャンセルされたか。
    """
    if not nodes or is_canceled():
        return

    rt = _rt()
    match modality:
        case Modality.TEXT:
            pipe = _build_text_pipeline(persist_dir)
        case Modality.IMAGE:
            pipe = _build_image_pipeline(persist_dir)
        case Modality.AUDIO:
            pipe = _build_audio_pipeline(persist_dir)
        case _:
            raise ValueError(f"unexpected modality: {modality}")

    total_batches = (len(nodes) + batch_size - 1) // batch_size
    trans_nodes = []
    for idx in range(0, len(nodes), batch_size):
        if is_canceled():
            logger.info("Job is canceled, aborting batch processing")
            return

        batch = nodes[idx : idx + batch_size]
        prog = f"{idx // batch_size + 1}/{total_batches}"
        logger.debug(
            f"{modality} upsert pipeline: processing batch {prog} "
            f"({len(batch)} nodes)"
        )
        try:
            trans_nodes.extend(await pipe.arun(nodes=batch))
        except Exception as e:
            logger.error(f"failed to process batch {prog}, continue: {e}")

    rt.persist_pipeline(pipe=pipe, modality=modality, persist_dir=persist_dir)
    logger.debug(f"{len(nodes)} nodes --pipeline--> {len(trans_nodes)} nodes")


async def _aupsert_nodes(
    text_nodes: Sequence[TextNode],
    image_nodes: Sequence[ImageNode],
    audio_nodes: Sequence[AudioNode],
    persist_dir: Optional[Path],
    batch_size: int,
    is_canceled: Callable[[], bool],
) -> None:
    """ノードをアップサートする。

    Args:
        text_nodes (Sequence[TextNode]): テキストノード
        image_nodes (Sequence[ImageNode]): 画像ノード
        audio_nodes (Sequence[AudioNode]): 音声ノード
        persist_dir (Optional[Path]): 永続化ディレクトリ
        batch_size (int): バッチサイズ
        is_canceled (Callable[[], bool]): このジョブがキャンセルされたか。
    """
    import asyncio

    tasks = []
    tasks.append(
        _process_batches(
            nodes=text_nodes,
            modality=Modality.TEXT,
            persist_dir=persist_dir,
            batch_size=batch_size,
            is_canceled=is_canceled,
        )
    )
    tasks.append(
        _process_batches(
            nodes=image_nodes,
            modality=Modality.IMAGE,
            persist_dir=persist_dir,
            batch_size=batch_size,
            is_canceled=is_canceled,
        )
    )
    tasks.append(
        _process_batches(
            nodes=audio_nodes,
            modality=Modality.AUDIO,
            persist_dir=persist_dir,
            batch_size=batch_size,
            is_canceled=is_canceled,
        )
    )

    await asyncio.gather(*tasks)

    _cleanup_temp_files()


def _cleanup_temp_files() -> None:
    """プレフィックスにマッチする一時ファイルをまとめて削除する。

    思わぬ取りこぼしがあるかもしれないので、ノードから一時ファイル名を取ることはしない。
    """
    import tempfile
    from pathlib import Path

    from ..core.const import TEMP_FILE_PREFIX

    temp_dir = Path(tempfile.gettempdir())
    prefix = TEMP_FILE_PREFIX

    try:
        entries = list(temp_dir.iterdir())
    except OSError as e:
        logger.warning(f"failed to list temp dir {temp_dir}: {e}")
        return

    for entry in entries:
        if not entry.is_file():
            continue

        if not entry.name.startswith(prefix):
            continue

        try:
            entry.unlink()
        except OSError as e:
            logger.warning(f"failed to remove temp file {entry}: {e}")


def ingest_path(
    path: str,
    batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """ローカルパス（ディレクトリ、ファイル）からコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        path (str): 対象パス
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
        is_canceled (Callable[[], bool], optional):
            このジョブがキャンセルされたか。Defaults to lambda:False.
    """
    asyncio_run(aingest_path(path, batch_size=batch_size, is_canceled=is_canceled))


async def aingest_path(
    path: str,
    batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """ローカルパス（ディレクトリ、ファイル）から非同期でコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        path (str): 対象パス
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
        is_canceled (Callable[[], bool], optional):
            このジョブがキャンセルされたか。Defaults to lambda:False.
    """
    rt = _rt()
    file_loader = rt.file_loader
    text_nodes, image_nodes, audio_nodes = await file_loader.aload_from_path(
        root=path, is_canceled=is_canceled
    )
    batch_size = batch_size or rt.cfg.ingest.batch_size

    await _aupsert_nodes(
        text_nodes=text_nodes,
        image_nodes=image_nodes,
        audio_nodes=audio_nodes,
        persist_dir=rt.cfg.ingest.pipe_persist_dir,
        batch_size=batch_size,
        is_canceled=is_canceled,
    )


def ingest_path_list(
    lst: str | Sequence[str],
    batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """パスリスト内の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        lst (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
        is_canceled (Callable[[], bool], optional):
            このジョブがキャンセルされたか。Defaults to lambda:False.
    """
    asyncio_run(aingest_path_list(lst, batch_size=batch_size, is_canceled=is_canceled))


async def aingest_path_list(
    lst: str | Sequence[str],
    batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """パスリスト内の複数パスから非同期でコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
        is_canceled (Callable[[], bool], optional):
            このジョブがキャンセルされたか。Defaults to lambda:False.
    """
    if isinstance(lst, str):
        lst = _read_list(lst)

    rt = _rt()
    file_loader = rt.file_loader
    text_nodes, image_nodes, audio_nodes = await file_loader.aload_from_paths(
        paths=list(lst), is_canceled=is_canceled
    )
    batch_size = batch_size or rt.cfg.ingest.batch_size

    await _aupsert_nodes(
        text_nodes=text_nodes,
        image_nodes=image_nodes,
        audio_nodes=audio_nodes,
        persist_dir=rt.cfg.ingest.pipe_persist_dir,
        batch_size=batch_size,
        is_canceled=is_canceled,
    )


def ingest_url(
    url: str,
    batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """URL からコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        url (str): 対象 URL
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
        is_canceled (Callable[[], bool], optional):
            このジョブがキャンセルされたか。Defaults to lambda:False.
    """
    asyncio_run(aingest_url(url=url, batch_size=batch_size, is_canceled=is_canceled))


async def aingest_url(
    url: str,
    batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """URL から非同期でコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        url (str): 対象 URL
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
        is_canceled (Callable[[], bool], optional):
            このジョブがキャンセルされたか。Defaults to lambda:False.
    """
    rt = _rt()
    html_loader = rt.html_loader
    text_nodes, image_nodes, audio_nodes = await html_loader.aload_from_url(
        url=url, is_canceled=is_canceled
    )
    batch_size = batch_size or rt.cfg.ingest.batch_size

    await _aupsert_nodes(
        text_nodes=text_nodes,
        image_nodes=image_nodes,
        audio_nodes=audio_nodes,
        persist_dir=rt.cfg.ingest.pipe_persist_dir,
        batch_size=batch_size,
        is_canceled=is_canceled,
    )


def ingest_url_list(
    lst: str | Sequence[str],
    batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """URL リスト内の複数サイトからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        lst (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
        is_canceled (Callable[[], bool], optional):
            このジョブがキャンセルされたか。Defaults to lambda:False.
    """
    asyncio_run(aingest_url_list(lst, batch_size=batch_size, is_canceled=is_canceled))


async def aingest_url_list(
    lst: str | Sequence[str],
    batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """URL リスト内の複数サイトから非同期でコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        lst (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
        is_canceled (Callable[[], bool], optional):
            このジョブがキャンセルされたか。Defaults to lambda:False.
    """
    if isinstance(lst, str):
        lst = _read_list(lst)

    rt = _rt()
    html_loader = rt.html_loader
    text_nodes, image_nodes, audio_nodes = await html_loader.aload_from_urls(
        urls=list(lst), is_canceled=is_canceled
    )
    batch_size = batch_size or rt.cfg.ingest.batch_size

    await _aupsert_nodes(
        text_nodes=text_nodes,
        image_nodes=image_nodes,
        audio_nodes=audio_nodes,
        persist_dir=rt.cfg.ingest.pipe_persist_dir,
        batch_size=batch_size,
        is_canceled=is_canceled,
    )
