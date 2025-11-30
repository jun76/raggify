from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict

from llama_index.core.schema import NodeWithScore


class FakeReranker:
    def __init__(self, *, model: str = "mock-model") -> None:
        self.model = model
        self.top_n = 1
        self.calls: list[dict[str, Any]] = []

    async def apostprocess_nodes(
        self, nodes: list[NodeWithScore], query_str: str
    ) -> list[NodeWithScore]:
        self.calls.append({"nodes": nodes, "query": query_str})
        return nodes[: self.top_n]


def setup_rerank_mocks(monkeypatch) -> DefaultDict[str, list[FakeReranker]]:
    records: DefaultDict[str, list[FakeReranker]] = defaultdict(list)

    def make_factory(key: str) -> Callable[..., FakeReranker]:
        def factory(*_, **kwargs) -> FakeReranker:
            inst = FakeReranker(model=kwargs.get("model", "mock-model"))
            records[key].append(inst)
            return inst

        return factory

    monkeypatch.setattr(
        "llama_index.postprocessor.cohere_rerank.CohereRerank",
        make_factory("cohere"),
    )
    monkeypatch.setattr(
        "llama_index.postprocessor.flag_embedding_reranker.FlagEmbeddingReranker",
        make_factory("flag"),
    )
    monkeypatch.setattr(
        "llama_index.postprocessor.voyageai_rerank.VoyageAIRerank",
        make_factory("voyage"),
    )

    return records
