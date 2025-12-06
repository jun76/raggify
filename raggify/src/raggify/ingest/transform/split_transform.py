from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Type

from llama_index.core.schema import BaseNode, TransformComponent

from ...config.ingest_config import IngestConfig
from ...core.const import PKG_NOT_FOUND_MSG
from ...core.event import async_loop_runner
from ...logger import logger

if TYPE_CHECKING:
    from llama_index.core.schema import TextNode

__all__ = ["SplitTransform"]


class SplitTransform(TransformComponent):
    """Base class for splitting media nodes into fixed-length chunks."""

    def __init__(
        self, cfg: IngestConfig, text_chunk_size: Optional[int] = None
    ) -> None:
        """Constructor.

        Args:
            cfg (IngestConfig): Ingest configuration.
            text_chunk_size (Optional[int]): Optional text chunk size to override config.
        """
        self._text_chunk_size = text_chunk_size or cfg.text_chunk_size
        self._text_chunk_overlap = cfg.text_chunk_overlap
        self._audio_chunk_seconds = cfg.audio_chunk_seconds
        self._video_chunk_seconds = cfg.video_chunk_seconds
        self._text_split_transform = None

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Synchronous interface.

        Args:
            nodes (list[BaseNode]): Nodes to summarize.

        Returns:
            list[BaseNode]: Nodes after summarization.
        """
        return async_loop_runner.run(lambda: self.acall(nodes=nodes, **kwargs))

    async def acall(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Interface called from the pipeline asynchronously.

        Args:
            nodes (list[BaseNode]): Nodes to split.

        Returns:
            list[BaseNode]: Nodes after splitting.
        """
        from llama_index.core.schema import TextNode

        from ...llama_like.core.schema import AudioNode, VideoNode

        split_nodes: list[BaseNode] = []
        for node in nodes:
            if isinstance(node, AudioNode):
                split = self._split_media(
                    node=node,
                    chunk_seconds=self._audio_chunk_seconds,
                    node_cls=AudioNode,
                )
            elif isinstance(node, VideoNode):
                split = self._split_media(
                    node=node,
                    chunk_seconds=self._video_chunk_seconds,
                    node_cls=VideoNode,
                )
            elif isinstance(node, TextNode):
                split = self._split_text(node)
            else:
                raise ValueError(f"unsupported node type: {type(node)}")

            split_nodes.extend(split)

        return split_nodes

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__

    def to_dict(self, **kwargs) -> dict:
        """Return a dict for caching that includes parameters."""

        return {
            "class_name": self.class_name(),
            "text_chunk_size": self._text_chunk_size,
            "text_chunk_overlap": self._text_chunk_overlap,
            "audio_chunk_seconds": self._audio_chunk_seconds,
            "video_chunk_seconds": self._video_chunk_seconds,
        }

    def _split_media(
        self, node: TextNode, chunk_seconds: Optional[int], node_cls: Type[TextNode]
    ) -> list[BaseNode]:
        """Split a single media node into multiple segments.

        Args:
            node (TextNode): Target node.
            chunk_seconds (Optional[int]): Chunk length in seconds.
            node_cls (Type[TextNode]): Node class to instantiate.

        Returns:
            list[BaseNode]: Split nodes or the original node on failure.
        """
        from ...core.metadata import MetaKeys as MK

        nodes: list[BaseNode] = [node]

        path = node.metadata.get(MK.FILE_PATH) or node.metadata.get(MK.TEMP_FILE_PATH)
        if not path:
            return nodes

        if chunk_seconds is None:
            return nodes

        duration = self._probe_duration(path)
        if duration is None or duration <= chunk_seconds:
            return nodes

        chunk_paths = self._create_segments(path=path, chunk_seconds=chunk_seconds)
        if not chunk_paths:
            return nodes

        return self._build_chunk_nodes(node, chunk_paths, node_cls)

    def _split_text(self, node: TextNode) -> list[BaseNode]:
        """Split a text node into smaller chunks.

        Args:
            node (TextNode): Target node.

        Returns:
            list[BaseNode]: Split nodes or the original node on failure.
        """
        from llama_index.core.node_parser import SentenceSplitter

        if self._text_split_transform is None:
            self._text_split_transform = SentenceSplitter(
                chunk_size=self._text_chunk_size,
                chunk_overlap=self._text_chunk_overlap,
                include_metadata=True,
            )

        try:
            return self._text_split_transform([node])
        except Exception as e:
            logger.warning(f"failed to split text node: {e}")
            return [node]

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

    def _create_segments(self, path: str, chunk_seconds: int) -> list[str]:
        """Create chunked files using ffmpeg.

        Args:
            path (str): Original media path.
            chunk_seconds (int): Chunk length in seconds.

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
                    segment_time=str(chunk_seconds),
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
