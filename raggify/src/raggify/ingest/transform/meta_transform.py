from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

from llama_index.core.schema import BaseNode

from ...logger import logger
from .base_transform import BaseTransform

__all__ = ["AddChunkIndexTransform", "RemoveTempFileTransform"]


class AddChunkIndexTransform(BaseTransform):
    """Transform to assign chunk indexes."""

    def __init__(self, is_canceled: Callable[[], bool]) -> None:
        """Constructor.

        Args:
            is_canceled (Callable[[], bool]): Cancellation flag for the job.
        """
        super().__init__(is_canceled)

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """Interface called from the pipeline.

        Args:
            nodes (Sequence[BaseNode]): Nodes already split.

        Returns:
            Sequence[BaseNode]: Nodes with chunk numbers assigned.
        """
        from ...core.metadata import MetaKeys as MK

        if not nodes:
            return nodes

        if self._is_canceled():
            logger.info("Job is canceled, aborting batch processing")
            return []

        node: BaseNode
        buckets = defaultdict(list)
        for node in nodes:
            id = node.ref_doc_id
            buckets[id].append(node)

        for id, group in buckets.items():
            for i, node in enumerate(group):
                node.metadata[MK.CHUNK_NO] = i

        if self._pipe_callback:
            self._pipe_callback(self, nodes)

        return nodes

    async def acall(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        return self.__call__(nodes, **kwargs)


class RemoveTempFileTransform(BaseTransform):
    """Transform to remove temporary files from nodes."""

    def __init__(self, is_canceled: Callable[[], bool]) -> None:
        """Constructor.

        Args:
            is_canceled (Callable[[], bool]): Cancellation flag for the job.
        """
        super().__init__(is_canceled)

    @classmethod
    def class_name(cls) -> str:
        """Return class name string.

        Returns:
            str: Class name.
        """
        return cls.__name__

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """Interface called from the pipeline.

        Args:
            nodes (Sequence[BaseNode]): Nodes to process.

        Returns:
            Sequence[BaseNode]: Nodes after removing temporary files.
        """
        import os

        from ...core.metadata import MetaKeys as MK

        if not nodes:
            return nodes

        if self._is_canceled():
            logger.info("Job is canceled, but continuing to remove temp files")

        for node in nodes:
            meta = node.metadata
            temp_file_path = meta.get(MK.TEMP_FILE_PATH)
            if temp_file_path:
                if os.path.exists(temp_file_path):
                    try:
                        os.remove(temp_file_path)
                    except Exception:
                        logger.warning(
                            f"failed to remove temporary file: {temp_file_path}"
                        )

                # Overwrite file_path with base_source for nodes with temp files
                # (either becomes empty or restores original path kept by
                # custom readers such as PDF)
                meta[MK.FILE_PATH] = meta[MK.BASE_SOURCE]

        if self._pipe_callback:
            self._pipe_callback(self, nodes)

        return nodes

    async def acall(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        return self.__call__(nodes, **kwargs)
