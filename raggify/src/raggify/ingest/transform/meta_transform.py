from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from llama_index.core.schema import BaseNode, TransformComponent

from ...logger import logger

__all__ = ["AddChunkIndexTransform", "RemoveTempFileTransform"]


class AddChunkIndexTransform(TransformComponent):
    """Transform to assign chunk indexes."""

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

        node: BaseNode
        buckets = defaultdict(list)
        for node in nodes:
            id = node.ref_doc_id
            buckets[id].append(node)

        for id, group in buckets.items():
            for i, node in enumerate(group):
                node.metadata[MK.CHUNK_NO] = i

        return nodes

    async def acall(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        return self.__call__(nodes, **kwargs)


class RemoveTempFileTransform(TransformComponent):
    """Transform to remove temporary files from nodes."""

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

        return nodes
