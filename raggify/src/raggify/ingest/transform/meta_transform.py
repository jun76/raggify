from __future__ import annotations

from collections import defaultdict

from llama_index.core.schema import BaseNode, TransformComponent

__all__ = ["AddChunkIndexTransform"]


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
        from ...core.metadata import MetaKeys as MK

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
