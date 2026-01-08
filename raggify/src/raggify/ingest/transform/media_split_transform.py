from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Type

from llama_index.core.schema import BaseNode

from ...config.ingest_config import IngestConfig
from ...core.event import async_loop_runner
from ...logger import logger
from .base_transform import BaseTransform

if TYPE_CHECKING:
    from llama_index.core.schema import TextNode

__all__ = ["MediaSplitTransform"]


class MediaSplitTransform(BaseTransform):
    """Transform to split media nodes into fixed-length chunks."""

    def __init__(self, cfg: IngestConfig, is_canceled: Callable[[], bool]) -> None:
        """Constructor.

        Args:
            cfg (IngestConfig): Ingest configuration.
            is_canceled (Callable[[], bool]): Cancellation flag for the job.
        """
        super().__init__(is_canceled)
        self._audio_chunk_seconds = cfg.audio_chunk_seconds
        self._video_chunk_seconds = cfg.video_chunk_seconds

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """Interface called from the pipeline.

        Args:
            nodes (Sequence[BaseNode]): Nodes to split.

        Returns:
            Sequence[BaseNode]: Nodes after splitting.
        """
        return async_loop_runner.run(lambda: self.acall(nodes=nodes, **kwargs))

    async def acall(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """Interface called from the pipeline asynchronously.

        Args:
            nodes (Sequence[BaseNode]): Nodes to split.

        Returns:
            Sequence[BaseNode]: Nodes after splitting.
        """
        from ...llama_like.core.schema import AudioNode, VideoNode

        if not nodes:
            return nodes

        split_nodes: list[BaseNode] = []
        for node in nodes:
            if self._is_canceled():
                logger.info("Job is canceled, aborting batch processing")
                return []

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
            else:
                raise ValueError(f"unsupported node type: {type(node)}")

            split_nodes.extend(split)

        if self._record_nodes:
            self._record_nodes(self, nodes)

        logger.debug(f"split into {len(split_nodes)} nodes")

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
        from ..util import MediaConverter

        path = node.metadata.get(MK.FILE_PATH) or node.metadata.get(MK.TEMP_FILE_PATH)
        if path is None:
            return [node]

        if chunk_seconds is None:
            return [node]

        base_dir = MediaConverter().split(src=Path(path), chunk_seconds=chunk_seconds)
        if base_dir is None:
            return [node]

        return self._build_chunk_nodes(node, base_dir, node_cls)

    def _build_chunk_nodes(
        self, node: TextNode, base_dir: Path, node_cls: Type[TextNode]
    ) -> list[BaseNode]:
        """Build chunk nodes from paths.

        Args:
            node (TextNode): Original node.
            base_dir (Path): Directory containing chunk files.
            node_cls (Type[TextNode]): Node class to instantiate.

        Returns:
            list[BaseNode]: List of new chunk nodes.
        """
        from ...core.metadata import BasicMetaData
        from ...core.metadata import MetaKeys as MK

        nodes: list[BaseNode] = []

        for index, chunk_path in enumerate(base_dir.iterdir()):
            meta = BasicMetaData()
            meta.file_path = str(chunk_path)
            meta.url = node.metadata.get(MK.URL, "")
            meta.temp_file_path = str(chunk_path)
            meta.base_source = node.metadata.get(MK.BASE_SOURCE, "")
            meta.chunk_no = index

            nodes.append(
                node_cls(text=node.text, id_=str(chunk_path), metadata=meta.to_dict())
            )

        logger.debug(f"split node into {len(nodes)} chunks under {base_dir}")

        return nodes
