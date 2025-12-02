from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type

from llama_index.core.schema import BaseNode, TransformComponent

from ...core.const import PKG_NOT_FOUND_MSG
from ...logger import logger

if TYPE_CHECKING:
    from llama_index.core.schema import TextNode

__all__ = ["AudioSplitter", "VideoSplitter"]


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
        from ...core.metadata import BasicMetaData
        from ...core.metadata import MetaKeys as MK

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
        from ...core.metadata import MetaKeys as MK
        from ...llama_like.core.schema import AudioNode, VideoNode

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

        Raises:
            ImportError: If ffmpeg is not installed.

        Returns:
            Optional[float]: Duration in seconds, or None on failure.
        """
        try:
            import ffmpeg  # type: ignore
        except ImportError:
            raise ImportError(
                PKG_NOT_FOUND_MSG.format(
                    pkg="ffmpeg-python (additionally, ffmpeg itself must be installed separately)",
                    extra="ffmpeg",
                    feature="ffmpeg",
                )
            )

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

        Raises:
            ImportError: If ffmpeg is not installed.

        Returns:
            list[str]: Paths to chunk files.
        """
        try:
            import ffmpeg  # type: ignore
        except ImportError:
            raise ImportError(
                PKG_NOT_FOUND_MSG.format(
                    pkg="ffmpeg-python (additionally, ffmpeg itself must be installed separately)",
                    extra="ffmpeg",
                    feature="ffmpeg",
                )
            )

        from ...core.utils import get_temp_file_path_from

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
        from ...llama_like.core.schema import AudioNode

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
        from ...llama_like.core.schema import VideoNode

        return isinstance(node, VideoNode)

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__
