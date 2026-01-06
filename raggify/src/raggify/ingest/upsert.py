from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from ..llama_like.core.schema import Modality
from ..logger import logger
from ..runtime import get_runtime

if TYPE_CHECKING:
    from llama_index.core.schema import (
        BaseNode,
        ImageNode,
        TextNode,
        TransformComponent,
    )

    from ..llama_like.core.schema import AudioNode, VideoNode
    from ..pipeline.pipeline_manager import TracablePipeline

__all__ = ["aupsert_nodes"]


def _build_text_pipeline(is_canceled: Callable[[], bool]) -> TracablePipeline:
    """Build an ingestion pipeline for text.

    Args:
        is_canceled (Callable[[], bool]): Cancellation flag for the job.

    Returns:
        TracablePipeline: Pipeline instance.
    """
    from .transform import (
        AddChunkIndexTransform,
        EmbedTransform,
        RemoveTempFileTransform,
    )

    rt = get_runtime()
    transformations: list[TransformComponent] = [
        AddChunkIndexTransform(is_canceled),
        EmbedTransform(embed=rt.embed_manager, is_canceled=is_canceled),
        RemoveTempFileTransform(is_canceled),
    ]

    return rt.pipeline.build(modality=Modality.TEXT, transformations=transformations)


def _build_image_pipeline(is_canceled: Callable[[], bool]) -> TracablePipeline:
    """Build an ingestion pipeline for images.

    Args:
        is_canceled (Callable[[], bool]): Cancellation flag for the job.

    Returns:
        TracablePipeline: Pipeline instance.
    """
    from .transform import (
        DefaultCaptionTransform,
        EmbedTransform,
        LLMCaptionTransform,
        RemoveTempFileTransform,
    )

    rt = get_runtime()
    transformations: list[TransformComponent] = [
        (
            LLMCaptionTransform(llm_manager=rt.llm_manager, is_canceled=is_canceled)
            if rt.cfg.general.image_caption_transform_provider is not None
            else DefaultCaptionTransform()
        ),
        EmbedTransform(embed=rt.embed_manager, is_canceled=is_canceled),
        RemoveTempFileTransform(is_canceled),
    ]

    return rt.pipeline.build(modality=Modality.IMAGE, transformations=transformations)


def _build_audio_pipeline(is_canceled: Callable[[], bool]) -> TracablePipeline:
    """Build an ingestion pipeline for audio.

    Args:
        is_canceled (Callable[[], bool]): Cancellation flag for the job.

    Returns:
        TracablePipeline: Pipeline instance.
    """
    from .transform import (
        DefaultCaptionTransform,
        EmbedTransform,
        LLMCaptionTransform,
        MediaSplitTransform,
        RemoveTempFileTransform,
    )

    rt = get_runtime()
    transformations: list[TransformComponent] = [
        (
            LLMCaptionTransform(llm_manager=rt.llm_manager, is_canceled=is_canceled)
            if rt.cfg.general.audio_caption_transform_provider is not None
            else DefaultCaptionTransform()
        ),
        MediaSplitTransform(cfg=rt.cfg.ingest, is_canceled=is_canceled),
    ]
    transformations.append(
        EmbedTransform(embed=rt.embed_manager, is_canceled=is_canceled)
    )
    transformations.append(RemoveTempFileTransform(is_canceled))

    return rt.pipeline.build(modality=Modality.AUDIO, transformations=transformations)


def _build_video_pipeline(is_canceled: Callable[[], bool]) -> TracablePipeline:
    """Build an ingestion pipeline for video.

    Args:
        is_canceled (Callable[[], bool]): Cancellation flag for the job.

    Returns:
        TracablePipeline: Pipeline instance.
    """
    from .transform import (
        DefaultCaptionTransform,
        EmbedTransform,
        LLMCaptionTransform,
        MediaSplitTransform,
        RemoveTempFileTransform,
    )

    rt = get_runtime()
    transformations: list[TransformComponent] = [
        (
            LLMCaptionTransform(llm_manager=rt.llm_manager, is_canceled=is_canceled)
            if rt.cfg.general.video_caption_transform_provider is not None
            else DefaultCaptionTransform()
        ),
        MediaSplitTransform(cfg=rt.cfg.ingest, is_canceled=is_canceled),
    ]
    transformations.append(
        EmbedTransform(embed=rt.embed_manager, is_canceled=is_canceled)
    )
    transformations.append(RemoveTempFileTransform(is_canceled))

    return rt.pipeline.build(modality=Modality.VIDEO, transformations=transformations)


def _build_pipeline(
    modality: Modality, is_canceled: Callable[[], bool]
) -> TracablePipeline:
    """Build an ingestion pipeline for a given modality.

    Args:
        modality (Modality): Target modality.
        is_canceled (Callable[[], bool]): Cancellation flag for the job.

    Returns:
        TracablePipeline: Pipeline instance.
    """
    match modality:
        case Modality.TEXT:
            return _build_text_pipeline(is_canceled)
        case Modality.IMAGE:
            return _build_image_pipeline(is_canceled)
        case Modality.AUDIO:
            return _build_audio_pipeline(is_canceled)
        case Modality.VIDEO:
            return _build_video_pipeline(is_canceled)
        case _:
            raise ValueError(f"unexpected modality: {modality}")


async def _process_batch(
    batch: Sequence[BaseNode],
    modality: Modality,
    persist_dir: Optional[Path],
    force: bool,
    pipe: TracablePipeline,
) -> Sequence[BaseNode]:
    """Process a batch of nodes through the pipeline.

    Args:
        batch (Sequence[BaseNode]): Nodes in the batch.
        modality (Modality): Target modality.
        persist_dir (Optional[Path]): Persist directory.
        force (bool): Whether to force reingestion even if already present.
        pipe (TracablePipeline): Pipeline instance.

    Raises:
        RuntimeError: If processing fails.

    Returns:
        Sequence[BaseNode]: Transformed nodes.
    """
    pipe.reset_nodes()

    rt = get_runtime()
    try:
        pipe.disable_cache = force
        transformed_nodes = await pipe.arun(nodes=batch)
        rt.pipeline.persist(pipe=pipe, persist_dir=persist_dir)

        # Return [] if no nodes were processed
        return transformed_nodes
    except Exception as e:
        # Roll back to prevent the next transform from being skipped
        # due to docstore duplicate detection.
        rt.document_store.delete_nodes(
            ref_doc_ids={
                node.ref_doc_id for node in batch if node.ref_doc_id is not None
            },
            persist_dir=persist_dir,
        )

        # Roll back cache entries
        for transformation, nodes in pipe.nodes:
            rt.ingest_cache.delete_nodes(
                modality=modality,
                nodes=nodes,
                transformation=transformation,
                persist_dir=persist_dir,
            )

        raise RuntimeError(
            f"failed to process {modality} batch, rolled back: {e}"
        ) from e
    finally:
        pipe.disable_cache = False


async def _process_batches(
    nodes: Sequence[BaseNode],
    modality: Modality,
    persist_dir: Optional[Path],
    pipe_batch_size: int,
    force: bool,
    is_canceled: Callable[[], bool],
    tree_nodes: Optional[Sequence[BaseNode]] = None,
) -> None:
    """Batch upserts to avoid long blocking when handling many nodes.

    Args:
        nodes (Sequence[BaseNode]): Nodes.
        modality (Modality): Target modality.
        persist_dir (Optional[Path]): Persist directory.
        pipe_batch_size (int): Number of nodes processed per pipeline batch.
        force (bool): Whether to force reingestion even if already present.
        is_canceled (Callable[[], bool]): Cancellation flag for the job.
        tree_nodes (Optional[Sequence[BaseNode]], optional):
            Tree nodes to add to the document store before processing. Defaults to None.
    """
    if not nodes or is_canceled():
        return

    pipe = _build_pipeline(modality=modality, is_canceled=is_canceled)

    if tree_nodes is not None and pipe.docstore is not None:
        pipe.docstore.add_documents(tree_nodes)

    total_batches = (len(nodes) + pipe_batch_size - 1) // pipe_batch_size
    logger.debug(
        f"starting {modality} upsert pipeline for "
        f"{len(nodes)} nodes in {total_batches} batches"
    )

    cfg = get_runtime().cfg.pipeline
    transformed = 0
    for idx in range(0, len(nodes), pipe_batch_size):
        retry_count = len(cfg.batch_retry_interval_sec)
        for i in range(retry_count):
            if is_canceled():
                logger.info("Job is canceled, aborting batch processing")
                return

            batch = nodes[idx : idx + pipe_batch_size]
            try:
                temp = await _process_batch(
                    batch=batch,
                    modality=modality,
                    persist_dir=persist_dir,
                    force=force,
                    pipe=pipe,
                )
                transformed += len(temp)
                await asyncio.sleep(cfg.batch_interval_sec)
                break
            except RuntimeError as e:
                logger.debug(f"retry {i + 1} / {retry_count}: {e}")

                await asyncio.sleep(cfg.batch_retry_interval_sec[i])
        else:
            logger.error(
                f"failed to process {modality} batch after {retry_count} attempts, aborting"
            )

    logger.debug(f"{len(nodes)} {modality} nodes --pipeline--> {transformed} nodes")


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


async def aupsert_nodes(
    text_tree_nodes: Sequence[BaseNode],
    text_leaf_nodes: Sequence[TextNode],
    image_nodes: Sequence[ImageNode],
    audio_nodes: Sequence[AudioNode],
    video_nodes: Sequence[VideoNode],
    persist_dir: Optional[Path],
    pipe_batch_size: int,
    force: bool,
    is_canceled: Callable[[], bool],
) -> None:
    """Upsert nodes into stores.

    Args:
        text_tree_nodes (Sequence[BaseNode]): Text tree nodes.
        text_leaf_nodes (Sequence[TextNode]): Text leaf nodes.
        image_nodes (Sequence[ImageNode]): Image nodes.
        audio_nodes (Sequence[AudioNode]): Audio nodes.
        video_nodes (Sequence[VideoNode]): Video nodes.
        persist_dir (Optional[Path]): Persist directory.
        pipe_batch_size (int): Number of nodes processed per pipeline batch.
        force (bool): Whether to force reingestion even if already present.
        is_canceled (Callable[[], bool]): Cancellation flag for the job.
    """
    rt = get_runtime()
    tasks = []

    if rt.cfg.general.text_embed_provider is not None:
        tasks.append(
            _process_batches(
                nodes=text_leaf_nodes,
                modality=Modality.TEXT,
                persist_dir=persist_dir,
                pipe_batch_size=pipe_batch_size,
                force=force,
                is_canceled=is_canceled,
                tree_nodes=text_tree_nodes,
            )
        )

    if rt.cfg.general.image_embed_provider is not None:
        tasks.append(
            _process_batches(
                nodes=image_nodes,
                modality=Modality.IMAGE,
                persist_dir=persist_dir,
                pipe_batch_size=pipe_batch_size,
                force=force,
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
                force=force,
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
                force=force,
                is_canceled=is_canceled,
            )
        )

    await asyncio.gather(*tasks)

    _cleanup_temp_files()
