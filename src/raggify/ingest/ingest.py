from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Sequence

from llama_index.core.async_utils import asyncio_run

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


async def _arun_pipe(
    nodes: Sequence[BaseNode],
    transformations: list[TransformComponent] | None,
    modality: Modality,
    persist_dir: Optional[Path],
) -> Optional[Sequence[BaseNode]]:
    """モダリティ別キャッシュ用パイプラインを実行する。

    Args:
        nodes (Sequence[BaseNode]): ノード
        transformations (list[TransformComponent] | None): 変換一式
        modality (Modality): モダリティ
        persist_dir (Optional[Path]): 永続化ディレクトリ

    Returns:
        Optional[Sequence[BaseNode]]: 重複ドキュメントフィルター後のノード
    """
    if not nodes:
        return None

    rt = _rt()
    pipe = rt.build_pipeline(
        transformations=transformations, modality=modality, persist_dir=persist_dir
    )
    filtered_nodes = await pipe.arun(nodes=nodes)
    rt.persist_pipeline(pipe=pipe, modality=modality, persist_dir=persist_dir)

    logger.debug(f"{len(nodes)} nodes --pipeline--> {len(filtered_nodes)} nodes")

    return filtered_nodes


async def _arun_text_pipe(
    nodes: Sequence[TextNode],
    persist_dir: Optional[Path],
) -> Optional[Sequence[BaseNode]]:
    """テキストノードのパイプラインを実行する。

    Args:
        nodes (Sequence[TextNode]): テキストノード
        persist_dir (Optional[Path]): 永続化ディレクトリ

    Returus:
        Optional[Sequence[BaseNode]]: 重複ドキュメントフィルター後のノード
    """
    from llama_index.core.node_parser import SentenceSplitter

    from .transform import AddChunkIndexTransform, make_text_embed_transform

    rt = _rt()
    return await _arun_pipe(
        nodes=nodes,
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
        persist_dir=persist_dir,
    )


async def _arun_image_pipe(
    nodes: Sequence[ImageNode],
    persist_dir: Optional[Path],
) -> Optional[Sequence[BaseNode]]:
    """画像ノードのパイプラインを実行する。

    Args:
        nodes (Sequence[ImageNode]): 画像ノード
        persist_dir (Optional[Path]): 永続化ディレクトリ

    Returus:
        Optional[Sequence[BaseNode]]: 重複ドキュメントフィルター後のノード
    """
    from .transform import make_image_embed_transform

    return await _arun_pipe(
        nodes=nodes,
        transformations=[
            make_image_embed_transform(_rt().embed_manager),
        ],
        modality=Modality.IMAGE,
        persist_dir=persist_dir,
    )


async def _arun_audio_pipe(
    nodes: Sequence[AudioNode],
    persist_dir: Optional[Path],
) -> Optional[Sequence[BaseNode]]:
    """音声ノードのパイプラインを実行する。

    Args:
        nodes (Sequence[AudioNode]): 音声ノード
        persist_dir (Optional[Path]): 永続化ディレクトリ

    Returus:
        Optional[Sequence[BaseNode]]: 重複ドキュメントフィルター後のノード
    """
    from .transform import make_audio_embed_transform

    return await _arun_pipe(
        nodes=nodes,
        transformations=[
            make_audio_embed_transform(_rt().embed_manager),
        ],
        modality=Modality.AUDIO,
        persist_dir=persist_dir,
    )


async def _process_batches(
    nodes: Sequence[BaseNode],
    runner: Callable[
        [Sequence, Optional[Path]], Awaitable[Optional[Sequence[BaseNode]]]
    ],
    label: str,
    persist_dir: Optional[Path],
    batch_size: int,
) -> None:
    """大量ノードのアップサートで長時間ブロックしないようにバッチ化する。

    Args:
        nodes (Sequence[BaseNode]): ノード
        runner (Callable[
            [Sequence, Optional[Path]], Awaitable[Optional[Sequence[BaseNode]]]
        ]): バッチ化対象の処理
        label (str): 経過表示用ラベル
        persist_dir (Optional[Path]): 永続化ディレクトリ
        batch_size (int): バッチサイズ
    """
    if not nodes:
        return

    total_batches = (len(nodes) + batch_size - 1) // batch_size
    for idx in range(0, len(nodes), batch_size):
        batch = nodes[idx : idx + batch_size]
        logger.debug(
            f"{label}: processing batch {idx // batch_size + 1}/{total_batches} "
            f"({len(batch)} nodes)"
        )
        await runner(batch, persist_dir)


async def _aupsert_nodes(
    text_nodes: Sequence[TextNode],
    image_nodes: Sequence[ImageNode],
    audio_nodes: Sequence[AudioNode],
    persist_dir: Optional[Path],
    batch_size: int,
):
    """ノードをアップサートする。

    Args:
        text_nodes (Sequence[TextNode]): テキストノード
        image_nodes (Sequence[ImageNode]): 画像ノード
        audio_nodes (Sequence[AudioNode]): 音声ノード
        persist_dir (Optional[Path]): 永続化ディレクトリ
        batch_size (int): バッチサイズ
    """
    import asyncio

    tasks = []
    tasks.append(
        _process_batches(
            nodes=text_nodes,
            runner=_arun_text_pipe,
            label="text pipeline",
            persist_dir=persist_dir,
            batch_size=batch_size,
        )
    )
    tasks.append(
        _process_batches(
            nodes=image_nodes,
            runner=_arun_image_pipe,
            label="image pipeline",
            persist_dir=persist_dir,
            batch_size=batch_size,
        )
    )
    tasks.append(
        _process_batches(
            nodes=audio_nodes,
            runner=_arun_audio_pipe,
            label="audio pipeline",
            persist_dir=persist_dir,
            batch_size=batch_size,
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
) -> None:
    """ローカルパス（ディレクトリ、ファイル）からコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        path (str): 対象パス
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
    """
    asyncio_run(aingest_path(path, batch_size=batch_size))


async def aingest_path(
    path: str,
    batch_size: Optional[int] = None,
) -> None:
    """ローカルパス（ディレクトリ、ファイル）から非同期でコンテンツを収集、埋め込み、ストアに格納する。
    ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

    Args:
        path (str): 対象パス
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
    """
    rt = _rt()
    file_loader = rt.file_loader
    text_nodes, image_nodes, audio_nodes = await file_loader.aload_from_path(path)
    batch_size = batch_size or rt.cfg.ingest.batch_size

    await _aupsert_nodes(
        text_nodes=text_nodes,
        image_nodes=image_nodes,
        audio_nodes=audio_nodes,
        persist_dir=rt.cfg.ingest.pipe_persist_dir,
        batch_size=batch_size,
    )


def ingest_path_list(
    lst: str | Sequence[str],
    batch_size: Optional[int] = None,
) -> None:
    """パスリスト内の複数パスからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        lst (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
    """
    asyncio_run(aingest_path_list(lst, batch_size=batch_size))


async def aingest_path_list(
    lst: str | Sequence[str],
    batch_size: Optional[int] = None,
) -> None:
    """パスリスト内の複数パスから非同期でコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        list (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
    """
    if isinstance(lst, str):
        lst = _read_list(lst)

    rt = _rt()
    file_loader = rt.file_loader
    text_nodes, image_nodes, audio_nodes = await file_loader.aload_from_paths(list(lst))
    batch_size = batch_size or rt.cfg.ingest.batch_size

    await _aupsert_nodes(
        text_nodes=text_nodes,
        image_nodes=image_nodes,
        audio_nodes=audio_nodes,
        persist_dir=rt.cfg.ingest.pipe_persist_dir,
        batch_size=batch_size,
    )


def ingest_url(
    url: str,
    batch_size: Optional[int] = None,
) -> None:
    """URL からコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        url (str): 対象 URL
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
    """
    asyncio_run(aingest_url(url, batch_size=batch_size))


async def aingest_url(
    url: str,
    batch_size: Optional[int] = None,
) -> None:
    """URL から非同期でコンテンツを収集、埋め込み、ストアに格納する。
    サイトマップ（.xml）の場合はツリーを下りながら複数サイトから取り込む。

    Args:
        url (str): 対象 URL
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
    """
    rt = _rt()
    html_loader = rt.html_loader
    text_nodes, image_nodes, audio_nodes = await html_loader.aload_from_url(url)
    batch_size = batch_size or rt.cfg.ingest.batch_size

    await _aupsert_nodes(
        text_nodes=text_nodes,
        image_nodes=image_nodes,
        audio_nodes=audio_nodes,
        persist_dir=rt.cfg.ingest.pipe_persist_dir,
        batch_size=batch_size,
    )


def ingest_url_list(
    lst: str | Sequence[str],
    batch_size: Optional[int] = None,
) -> None:
    """URL リスト内の複数サイトからコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        lst (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
    """
    asyncio_run(aingest_url_list(lst, batch_size=batch_size))


async def aingest_url_list(
    lst: str | Sequence[str],
    batch_size: Optional[int] = None,
) -> None:
    """URL リスト内の複数サイトから非同期でコンテンツを収集、埋め込み、ストアに格納する。

    Args:
        lst (str | Sequence[str]): テキストファイルまたは Sequence 形式のリスト
        batch_size (Optional[int]): バッチサイズ。Defaults to None.
    """
    if isinstance(lst, str):
        lst = _read_list(lst)

    rt = _rt()
    html_loader = rt.html_loader
    text_nodes, image_nodes, audio_nodes = await html_loader.aload_from_urls(list(lst))
    batch_size = batch_size or rt.cfg.ingest.batch_size

    await _aupsert_nodes(
        text_nodes=text_nodes,
        image_nodes=image_nodes,
        audio_nodes=audio_nodes,
        persist_dir=rt.cfg.ingest.pipe_persist_dir,
        batch_size=batch_size,
    )
