from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Type

from llama_index.core.async_utils import asyncio_run
from llama_index.core.schema import BaseNode, TransformComponent

from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import Embedding
    from llama_index.core.schema import ImageType, TextNode

    from ..embed.embed_manager import EmbedManager
    from ..llama.embeddings.multi_modal_base import AudioType, VideoType


class AddChunkIndexTransform(TransformComponent):
    """Transform to assign chunk indexes."""

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Interface called from the pipeline.

        Args:
            nodes (list[BaseNode]): Nodes already split.

        Returns:
            list[BaseNode]: Nodes with chunk numbers assigned.
        """
        from ..core.metadata import MetaKeys as MK

        buckets = defaultdict(list)
        for node in nodes:
            id = node.ref_doc_id
            buckets[id].append(node)

        node: BaseNode
        for id, group in buckets.items():
            for i, node in enumerate(group):
                node.metadata[MK.CHUNK_NO] = i

        return nodes

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        return self.__call__(nodes, **kwargs)


class _BaseMediaSplitter(TransformComponent):
    """Base class for splitting media nodes into fixed-length chunks."""

    def __init__(self, chunk_seconds: Optional[int] = None) -> None:
        """Constructor.

        Args:
            chunk_seconds (Optional[int], optional): Chunk length in seconds.
        """
        self._chunk_seconds = chunk_seconds or 0

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Split matching media nodes.

        Args:
            nodes (list[BaseNode]): Input nodes.

        Returns:
            list[BaseNode]: Nodes with split segments replacing originals.
        """
        if self._chunk_seconds <= 0:
            return nodes

        result: list[BaseNode] = []
        for node in nodes:
            if not self._matches(node):
                result.append(node)
                continue

            try:
                result.extend(self._split_node(node))
            except Exception as e:
                logger.warning(f"failed to split media node: {e}")
                result.append(node)

        return result

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Async wrapper matching the synchronous call.

        Args:
            nodes (list[BaseNode]): Input nodes.

        Returns:
            list[BaseNode]: Nodes with split segments replacing originals.
        """
        return self.__call__(nodes, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__

    def to_dict(self, **kwargs) -> dict:
        """Return a dict for caching that includes split seconds.

        Args:
            **kwargs: Unused additional options.

        Returns:
            dict: Serializable payload for hashing.
        """
        return {"class_name": self.class_name(), "chunk_seconds": self._chunk_seconds}

    def _matches(self, node: BaseNode) -> bool:
        """Return True if the node should be split."""
        raise NotImplementedError

    def _build_chunk_nodes(
        self, node: TextNode, chunk_paths: list[str], node_cls: Type[TextNode]
    ) -> list[BaseNode]:
        """Build chunk nodes from paths.

        Args:
            node (TextNode): Original node.
            chunk_paths (list[str]): Chunk file paths.
            node_cls (Type[TextNode]): Node class to instantiate.

        Returns:
            list[BaseNode]: List of new chunk nodes.
        """
        from ..core.metadata import BasicMetaData
        from ..core.metadata import MetaKeys as MK

        nodes: list[BaseNode] = []
        for index, chunk_path in enumerate(chunk_paths):
            meta = BasicMetaData()
            meta.file_path = chunk_path
            meta.url = node.metadata.get(MK.URL, "")
            meta.temp_file_path = chunk_path
            meta.base_source = node.metadata.get(MK.BASE_SOURCE, "")
            meta.chunk_no = index

            nodes.append(
                node_cls(
                    text=node.text,
                    id_=chunk_path,
                    ref_doc_id=node.ref_doc_id,
                    metadata=meta.to_dict(),
                )
            )

        return nodes

    def _split_node(self, node: BaseNode) -> list[BaseNode]:
        """Split a single node into multiple segments.

        Args:
            node (BaseNode): Target node.

        Returns:
            list[BaseNode]: Split nodes or the original node on failure.
        """
        from ..core.metadata import MetaKeys as MK
        from ..llama.core.schema import AudioNode, VideoNode

        nodes = [node]

        path = node.metadata.get(MK.FILE_PATH) or node.metadata.get(MK.TEMP_FILE_PATH)
        if not path:
            return nodes

        duration = self._probe_duration(path)
        if duration is None or duration <= self._chunk_seconds:
            return nodes

        chunk_paths = self._create_segments(path)
        if not chunk_paths:
            return nodes

        if isinstance(node, AudioNode):
            nodes = self._build_chunk_nodes(node, chunk_paths, AudioNode)
        elif isinstance(node, VideoNode):
            nodes = self._build_chunk_nodes(node, chunk_paths, VideoNode)
        else:
            logger.warning(f"unexpected node type {type(node)}, skipped")

        return nodes

    def _probe_duration(self, path: str) -> Optional[float]:
        """Inspect media duration via ffmpeg.

        Args:
            path (str): Media file path.

        Returns:
            Optional[float]: Duration in seconds, or None on failure.
        """
        import ffmpeg

        try:
            probe = ffmpeg.probe(path)
            return float(probe["format"]["duration"])
        except Exception as e:
            logger.warning(f"failed to probe media duration for {path}: {e}")
            return None

    def _create_segments(self, path: str) -> list[str]:
        """Create chunked files using ffmpeg.

        Args:
            path (str): Original media path.

        Returns:
            list[str]: Paths to chunk files.
        """
        import ffmpeg

        from ..core.utils import get_temp_file_path_from

        ext = Path(path).suffix
        base_path = Path(get_temp_file_path_from(source=path, suffix=ext))
        temp_dir = base_path.parent / f"{base_path.stem}_chunks"

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        temp_dir.mkdir(parents=True, exist_ok=True)
        pattern = temp_dir / f"{base_path.stem}_%05d{ext}"
        try:
            (
                ffmpeg.input(path)
                .output(
                    str(pattern),
                    f="segment",
                    segment_time=str(self._chunk_seconds),
                    c="copy",
                    reset_timestamps="1",
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logger.warning(f"ffmpeg split failed for {path}: {e}")
            return []

        chunks = sorted(str(p) for p in temp_dir.glob(f"{base_path.stem}_*{ext}"))
        logger.debug(f"split to {len(chunks)} chunk(s) from {path}")

        return chunks


class AudioSplitter(_BaseMediaSplitter):
    """Split audio nodes into smaller segments."""

    def __init__(self, chunk_seconds: Optional[int] = None) -> None:
        """Constructor.

        Args:
            chunk_seconds (Optional[int], optional): Chunk length in seconds.
        """
        super().__init__(chunk_seconds)

    def _matches(self, node: BaseNode) -> bool:
        from ..llama.core.schema import AudioNode

        return isinstance(node, AudioNode)

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__


class VideoSplitter(_BaseMediaSplitter):
    """Split video nodes into smaller segments."""

    def __init__(self, chunk_seconds: Optional[int] = None) -> None:
        """Constructor.

        Args:
            chunk_seconds (Optional[int], optional): Chunk length in seconds.
        """
        super().__init__(chunk_seconds)

    def _matches(self, node: BaseNode) -> bool:
        from ..llama.core.schema import VideoNode

        return isinstance(node, VideoNode)

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__


class _BaseEmbedTransform(TransformComponent):
    """Base transform for embedding."""

    def __init__(
        self,
        batch_embed_fn: Callable[[list], Awaitable[list[list[float]]]],
        extract_fn: Callable[[BaseNode], object],
        name: str = "",
    ):
        """Constructor.

        Args:
            batch_embed_fn (Callable[[list], Awaitable[list[list[float]]]]):
                Batch embedding function.
            extract_fn (Callable[[BaseNode], object]): Modality-specific extractor.
            name (str, optional): Name label for cache separation. Defaults to "".
        """
        self._batch_embed_fn = batch_embed_fn
        self._extract_fn = extract_fn
        self._name = name or self.__class__.__name__

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__

    def to_dict(self, **kwargs) -> dict:
        """Return a dict for caching that includes name label.

        Args:
            **kwargs: Unused additional options.

        Returns:
            dict: Serializable payload for hashing.
        """
        return {"class_name": self.class_name(), "name": self._name}

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Synchronous interface.

        Args:
            nodes (list[BaseNode]): Nodes to embed.

        Returns:
            list[BaseNode]: Nodes after embedding.
        """
        return asyncio_run(self.acall(nodes=nodes, **kwargs))

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Interface called from the pipeline asynchronously.

        Args:
            nodes (list[BaseNode]): Nodes to embed.

        Returns:
            list[BaseNode]: Nodes after embedding.
        """
        from ..core.metadata import MetaKeys as MK

        # Extract inputs (skip missing while keeping back-references to original nodes)
        inputs: list[object] = []
        backrefs: list[int] = []
        for i, node in enumerate(nodes):
            x = self._extract_fn(node)
            if x is None:
                continue

            inputs.append(x)
            backrefs.append(i)

        if not inputs:
            return nodes

        # Batch embedding
        vecs = await self._batch_embed_fn(inputs)
        if not vecs:
            return nodes

        if len(vecs) != len(inputs):
            # Safety: do not write when lengths differ (log at caller)
            return nodes

        # Write back to nodes
        for i, vec in zip(backrefs, vecs):
            nodes[i].embedding = vec

            if nodes[i].metadata.get(MK.TEMP_FILE_PATH):
                # Overwrite file_path with base_source for nodes with temp files
                # (either becomes empty or restores original path kept by
                # custom readers such as PDF)
                nodes[i].metadata[MK.FILE_PATH] = nodes[i].metadata[MK.BASE_SOURCE]

        return nodes


def _get_media_path(node: BaseNode) -> str:
    """Get media path for embedded non-text content.

    Args:
        node (BaseNode): Target node.

    Returns:
        str: Media path.
    """
    from ..core.metadata import MetaKeys as MK

    temp = node.metadata.get(MK.TEMP_FILE_PATH)
    if temp:
        # Temp file fetched
        return temp

    # Local file
    return node.metadata[MK.FILE_PATH]


def make_text_embed_transform(embed: EmbedManager) -> _BaseEmbedTransform:
    """Factory for text embedding transform.

    Args:
        embed (EmbedManager): Embedding manager.

    Returns:
        _BaseEmbedTransform: Transform instance.
    """
    from llama_index.core.schema import TextNode

    async def batch_text(texts: list[str]) -> list[Embedding]:
        return await embed.aembed_text(texts)

    def extractor(node: BaseNode) -> Optional[str]:
        if isinstance(node, TextNode) and node.text and node.text.strip():
            return node.text

        logger.warning("text is not found, skipped")
        return None

    return _BaseEmbedTransform(batch_text, extractor, name="text_embed")


def make_image_embed_transform(embed: EmbedManager) -> _BaseEmbedTransform:
    """Factory for image embedding transform.

    Args:
        embed (EmbedManager): Embedding manager.

    Returns:
        _BaseEmbedTransform: Transform instance.
    """
    from llama_index.core.schema import ImageNode

    async def batch_image(paths: list[ImageType]) -> list[Embedding]:
        return await embed.aembed_image(paths)

    def extractor(node: BaseNode) -> Optional[str]:
        if isinstance(node, ImageNode):
            return _get_media_path(node)

        logger.warning("image is not found, skipped")
        return None

    return _BaseEmbedTransform(batch_image, extractor, name="image_embed")


def make_audio_embed_transform(embed: EmbedManager) -> _BaseEmbedTransform:
    """Factory for audio embedding transform.

    Args:
        embed (EmbedManager): Embedding manager.

    Returns:
        _BaseEmbedTransform: Transform instance.
    """
    from ..core.exts import Exts
    from ..core.utils import has_media

    async def batch_audio(paths: list[AudioType]) -> list[Embedding]:
        return await embed.aembed_audio(paths)

    def extractor(node: BaseNode) -> Optional[str]:
        # Can't use isinstance because AudioNode is not known by llama_index
        if has_media(node=node, exts=Exts.AUDIO):
            return _get_media_path(node)

        logger.warning("audio is not found, skipped")
        return None

    return _BaseEmbedTransform(batch_audio, extractor, name="audio_embed")


def make_video_embed_transform(embed: EmbedManager) -> _BaseEmbedTransform:
    """Factory for video embedding transform.

    Args:
        embed (EmbedManager): Embedding manager.

    Returns:
        _BaseEmbedTransform: Transform instance.
    """
    from ..core.exts import Exts
    from ..core.utils import has_media

    async def batch_video(paths: list[VideoType]) -> list[Embedding]:
        return await embed.aembed_video(paths)

    def extractor(node: BaseNode) -> Optional[str]:
        # Can't use isinstance because VideoNode is not known by llama_index
        if has_media(node=node, exts=Exts.VIDEO):
            return _get_media_path(node)

        logger.warning("video is not found, skipped")
        return None

    return _BaseEmbedTransform(batch_video, extractor, name="video_embed")
