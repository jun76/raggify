from collections import defaultdict
from typing import List

from llama_index.core.schema import BaseNode, TransformComponent


class AddChunkIndexTransform(TransformComponent):
    def __call__(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        buckets = defaultdict(list)
        for n in nodes:
            doc_id = n.metadata.get("doc_id") or n.metadata.get("file_path")
            buckets[doc_id].append(n)

        for doc_id, group in buckets.items():
            total = len(group)
            for i, n in enumerate(group):
                n.metadata["chunk_index"] = i
                n.metadata["chunk_total"] = total
                # 任意: 安定IDを作るならここで
                # n.id_ = f"{doc_id}:{i}"  # 既存の node_id との整合は方針に合わせて
        return nodes

    async def acall(self, nodes: List[BaseNode], **kwargs) -> List[BaseNode]:
        return self.__call__(nodes, **kwargs)
