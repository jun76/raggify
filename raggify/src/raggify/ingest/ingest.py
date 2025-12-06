from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from llama_index.core.ingestion import IngestionPipeline

from ..core.event import async_loop_runner
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

    from ..llama_like.core.schema import AudioNode, VideoNode


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
    """Read a list of paths or URLs from a file.

    Args:
        path (str): Path to the list file.

    Returns:
        list[str]: Loaded list.
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
    """Build an ingestion pipeline for text.

    Args:
        persist_dir (Optional[Path]): Persist directory.

    Returns:
        IngestionPipeline: Pipeline instance.
    """
    from .transform import (
        AddChunkIndexTransform,
        DefaultSummarizeTransform,
        EmbedTransform,
        LLMSummarizeTransform,
        SplitTransform,
    )

    rt = _rt()
    if rt.cfg.general.text_summarize_transform_provider is not None:
        transformations: list[TransformComponent] = [
            # Split before LLM summarization to avoid token limit issues
            SplitTransform(cfg=rt.cfg.ingest, text_chunk_size=10000),
            LLMSummarizeTransform(rt.llm_manager),
        ]
    else:
        transformations: list[TransformComponent] = [
            DefaultSummarizeTransform(),
        ]

    transformations.append(SplitTransform(rt.cfg.ingest))
    transformations.append(AddChunkIndexTransform())
    transformations.append(EmbedTransform(rt.embed_manager))

    return rt.build_pipeline(
        modality=Modality.TEXT,
        transformations=transformations,
        persist_dir=persist_dir,
    )


def _build_image_pipeline(persist_dir: Optional[Path]) -> IngestionPipeline:
    """Build an ingestion pipeline for images.

    Args:
        persist_dir (Optional[Path]): Persist directory.

    Returns:
        IngestionPipeline: Pipeline instance.
    """
    from .transform import (
        DefaultSummarizeTransform,
        EmbedTransform,
        LLMSummarizeTransform,
    )

    rt = _rt()
    transformations: list[TransformComponent] = [
        (
            LLMSummarizeTransform(rt.llm_manager)
            if rt.cfg.general.image_summarize_transform_provider is not None
            else DefaultSummarizeTransform()
        ),
        EmbedTransform(rt.embed_manager),
    ]

    return rt.build_pipeline(
        modality=Modality.IMAGE,
        transformations=transformations,
        persist_dir=persist_dir,
    )


def _build_audio_pipeline(persist_dir: Optional[Path]) -> IngestionPipeline:
    """Build an ingestion pipeline for audio.

    Args:
        persist_dir (Optional[Path]): Persist directory.

    Returns:
        IngestionPipeline: Pipeline instance.
    """
    from .transform import (
        DefaultSummarizeTransform,
        EmbedTransform,
        LLMSummarizeTransform,
        SplitTransform,
    )

    rt = _rt()
    transformations: list[TransformComponent] = [
        (
            LLMSummarizeTransform(rt.llm_manager)
            if rt.cfg.general.audio_summarize_transform_provider is not None
            else DefaultSummarizeTransform()
        ),
        SplitTransform(rt.cfg.ingest),
    ]
    transformations.append(EmbedTransform(rt.embed_manager))
    return rt.build_pipeline(
        modality=Modality.AUDIO,
        transformations=transformations,
        persist_dir=persist_dir,
    )


def _build_video_pipeline(persist_dir: Optional[Path]) -> IngestionPipeline:
    """Build an ingestion pipeline for video.

    Args:
        persist_dir (Optional[Path]): Persist directory.

    Returns:
        IngestionPipeline: Pipeline instance.
    """
    from .transform import (
        DefaultSummarizeTransform,
        EmbedTransform,
        LLMSummarizeTransform,
        SplitTransform,
    )

    rt = _rt()
    transformations: list[TransformComponent] = [
        (
            LLMSummarizeTransform(rt.llm_manager)
            if rt.cfg.general.video_summarize_transform_provider is not None
            else DefaultSummarizeTransform()
        ),
        SplitTransform(rt.cfg.ingest),
    ]
    transformations.append(EmbedTransform(rt.embed_manager))

    return rt.build_pipeline(
        modality=Modality.VIDEO,
        transformations=transformations,
        persist_dir=persist_dir,
    )


async def _process_batches(
    nodes: Sequence[BaseNode],
    modality: Modality,
    persist_dir: Optional[Path],
    pipe_batch_size: int,
    is_canceled: Callable[[], bool],
    retry_count: int = 5,
) -> None:
    """Batch upserts to avoid long blocking when handling many nodes.

    Args:
        nodes (Sequence[BaseNode]): Nodes.
        modality (Modality): Target modality.
        persist_dir (Optional[Path]): Persist directory.
        pipe_batch_size (int): Number of nodes processed per pipeline batch.
        is_canceled (Callable[[], bool]): Cancellation flag for the job.
        retry_count (int): Number of retry attempts for processing a batch.
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
        case Modality.VIDEO:
            pipe = _build_video_pipeline(persist_dir)
        case _:
            raise ValueError(f"unexpected modality: {modality}")

    total_batches = (len(nodes) + pipe_batch_size - 1) // pipe_batch_size
    trans_nodes = []
    for idx in range(0, len(nodes), pipe_batch_size):
        delay = 1
        for i in range(retry_count):
            try:
                if is_canceled():
                    logger.info("Job is canceled, aborting batch processing")
                    return

                batch = nodes[idx : idx + pipe_batch_size]
                prog = f"{idx // pipe_batch_size + 1}/{total_batches}"
                logger.debug(
                    f"{modality} upsert pipeline: processing batch {prog} "
                    f"({len(batch)} nodes)"
                )

                trans_nodes.extend(await pipe.arun(nodes=batch))
                break
            except Exception as e:
                for node in batch:
                    if node.ref_doc_id is None:
                        continue

                    # Roll back to prevent the next transform from being skipped
                    # due to docstore duplicate detection.
                    rt.document_store.store.delete_ref_doc(
                        ref_doc_id=node.ref_doc_id, raise_error=False
                    )

                # FIXME: issue #4 Excessive rollbacks in ingest._process_batches

                # Roll back cache entries
                # rt.ingest_cache.delete(
                #     modality=modality,
                #     nodes=batch,
                #     transformations=pipe.transformations,
                #     persist_dir=persist_dir,
                # )
                rt.ingest_cache.delete_all(persist_dir)

                logger.error(f"failed to process batch {prog}, rolled back: {e}")
                time.sleep(delay)
                delay *= 2
                logger.debug(f"retry count: {i + 1} / {retry_count}")

    rt.persist_pipeline(pipe=pipe, modality=modality, persist_dir=persist_dir)
    logger.debug(f"{len(nodes)} nodes --pipeline--> {len(trans_nodes)} nodes")


async def _aupsert_nodes(
    text_nodes: Sequence[TextNode],
    image_nodes: Sequence[ImageNode],
    audio_nodes: Sequence[AudioNode],
    video_nodes: Sequence[VideoNode],
    persist_dir: Optional[Path],
    pipe_batch_size: int,
    is_canceled: Callable[[], bool],
) -> None:
    """Upsert nodes into stores.

    Args:
        text_nodes (Sequence[TextNode]): Text nodes.
        image_nodes (Sequence[ImageNode]): Image nodes.
        audio_nodes (Sequence[AudioNode]): Audio nodes.
        video_nodes (Sequence[VideoNode]): Video nodes.
        persist_dir (Optional[Path]): Persist directory.
        pipe_batch_size (int): Number of nodes processed per pipeline batch.
        is_canceled (Callable[[], bool]): Cancellation flag for the job.
    """
    import asyncio

    rt = _rt()
    tasks = []

    if rt.cfg.general.text_embed_provider is not None:
        tasks.append(
            _process_batches(
                nodes=text_nodes,
                modality=Modality.TEXT,
                persist_dir=persist_dir,
                pipe_batch_size=pipe_batch_size,
                is_canceled=is_canceled,
            )
        )

    if rt.cfg.general.image_embed_provider is not None:
        tasks.append(
            _process_batches(
                nodes=image_nodes,
                modality=Modality.IMAGE,
                persist_dir=persist_dir,
                pipe_batch_size=pipe_batch_size,
                is_canceled=is_canceled,
            )
        )

    if rt.cfg.general.audio_embed_provider is not None:
        tasks.append(
            _process_batches(
                nodes=audio_nodes,
                modality=Modality.AUDIO,
                persist_dir=persist_dir,
                pipe_batch_size=pipe_batch_size,
                is_canceled=is_canceled,
            )
        )

    if rt.cfg.general.video_embed_provider is not None:
        tasks.append(
            _process_batches(
                nodes=video_nodes,
                modality=Modality.VIDEO,
                persist_dir=persist_dir,
                pipe_batch_size=pipe_batch_size,
                is_canceled=is_canceled,
            )
        )

    await asyncio.gather(*tasks)

    _cleanup_temp_files()


def _cleanup_temp_files() -> None:
    """Remove temporary files that match the prefix.

    Avoid deriving names from nodes to prevent accidental misses.
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
        if not entry.name.startswith(prefix):
            continue

        try:
            if entry.is_dir():
                import shutil

                shutil.rmtree(entry)
            else:
                entry.unlink()
        except OSError as e:
            logger.warning(f"failed to remove temp entry {entry}: {e}")


def ingest_path(
    path: str,
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Ingest, embed, and store content from a local path (directory or file).

    Directories are traversed recursively to ingest multiple files.

    Args:
        path (str): Target path.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag. Defaults to lambda:False.
    """
    async_loop_runner.run(
        lambda: aingest_path(
            path, pipe_batch_size=pipe_batch_size, is_canceled=is_canceled
        )
    )


async def aingest_path(
    path: str,
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Asynchronously ingest, embed, and store content from a local path.

    Directories are traversed recursively to ingest multiple files.

    Args:
        path (str): Target path.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag. Defaults to lambda:False.
    """
    rt = _rt()
    texts, images, audios, videos = await rt.file_loader.aload_from_path(path)
    pipe_batch_size = pipe_batch_size or rt.cfg.ingest.pipe_batch_size

    await _aupsert_nodes(
        text_nodes=texts,
        image_nodes=images,
        audio_nodes=audios,
        video_nodes=videos,
        persist_dir=rt.cfg.ingest.pipe_persist_dir,
        pipe_batch_size=pipe_batch_size,
        is_canceled=is_canceled,
    )


def ingest_path_list(
    lst: str | Sequence[str],
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Ingest, embed, and store content from multiple paths in a list.

    Args:
        lst (str | Sequence[str]): Text file path or in-memory sequence.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag. Defaults to lambda:False.
    """
    async_loop_runner.run(
        lambda: aingest_path_list(
            lst, pipe_batch_size=pipe_batch_size, is_canceled=is_canceled
        )
    )


async def aingest_path_list(
    lst: str | Sequence[str],
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Asynchronously ingest, embed, and store content from multiple paths.

    Args:
        lst (str | Sequence[str]): Text file path or in-memory sequence.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag. Defaults to lambda:False.
    """
    if isinstance(lst, str):
        lst = _read_list(lst)

    rt = _rt()
    texts, images, audios, videos = await rt.file_loader.aload_from_paths(
        paths=list(lst), is_canceled=is_canceled
    )
    pipe_batch_size = pipe_batch_size or rt.cfg.ingest.pipe_batch_size
    await _aupsert_nodes(
        text_nodes=texts,
        image_nodes=images,
        audio_nodes=audios,
        video_nodes=videos,
        persist_dir=rt.cfg.ingest.pipe_persist_dir,
        pipe_batch_size=pipe_batch_size,
        is_canceled=is_canceled,
    )


def ingest_url(
    url: str,
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Ingest, embed, and store content from a URL.

    For sitemaps (.xml), traverse the tree to ingest multiple sites.

    Args:
        url (str): Target URL.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag. Defaults to lambda:False.
    """
    async_loop_runner.run(
        lambda: aingest_url(
            url=url, pipe_batch_size=pipe_batch_size, is_canceled=is_canceled
        )
    )


async def aingest_url(
    url: str,
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Asynchronously ingest, embed, and store content from a URL.

    For sitemaps (.xml), traverse the tree to ingest multiple sites.

    Args:
        url (str): Target URL.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag. Defaults to lambda:False.
    """
    rt = _rt()
    texts, images, audios, videos = await rt.web_page_loader.aload_from_url(
        url=url, is_canceled=is_canceled
    )
    pipe_batch_size = pipe_batch_size or rt.cfg.ingest.pipe_batch_size

    await _aupsert_nodes(
        text_nodes=texts,
        image_nodes=images,
        audio_nodes=audios,
        video_nodes=videos,
        persist_dir=rt.cfg.ingest.pipe_persist_dir,
        pipe_batch_size=pipe_batch_size,
        is_canceled=is_canceled,
    )


def ingest_url_list(
    lst: str | Sequence[str],
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Ingest, embed, and store content from multiple URLs in a list.

    Args:
        lst (str | Sequence[str]): Text file path or in-memory URL list.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag. Defaults to lambda:False.
    """
    async_loop_runner.run(
        lambda: aingest_url_list(
            lst, pipe_batch_size=pipe_batch_size, is_canceled=is_canceled
        )
    )


async def aingest_url_list(
    lst: str | Sequence[str],
    pipe_batch_size: Optional[int] = None,
    is_canceled: Callable[[], bool] = lambda: False,
) -> None:
    """Asynchronously ingest, embed, and store content from multiple URLs.

    Args:
        lst (str | Sequence[str]): Text file path or in-memory URL list.
        pipe_batch_size (Optional[int]):
            Number of nodes processed per pipeline batch. Defaults to None.
        is_canceled (Callable[[], bool], optional):
            Cancellation flag. Defaults to lambda:False.
    """
    if isinstance(lst, str):
        lst = _read_list(lst)

    rt = _rt()
    texts, images, audios, videos = await rt.web_page_loader.aload_from_urls(
        urls=list(lst), is_canceled=is_canceled
    )
    pipe_batch_size = pipe_batch_size or rt.cfg.ingest.pipe_batch_size

    await _aupsert_nodes(
        text_nodes=texts,
        image_nodes=images,
        audio_nodes=audios,
        video_nodes=videos,
        persist_dir=rt.cfg.ingest.pipe_persist_dir,
        pipe_batch_size=pipe_batch_size,
        is_canceled=is_canceled,
    )
