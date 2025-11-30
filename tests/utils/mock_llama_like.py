from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

from llama_index.core.schema import BaseNode, TextNode


class DummyEmbedModel:
    def __init__(self, values: Dict[str, Sequence[float]]):
        self._values = values

    async def aget_text_embedding_batch(
        self, texts: list[str], show_progress: bool = True
    ) -> list[Sequence[float]]:
        return [self._values["text"] for _ in texts]

    async def aget_audio_embedding_batch(
        self, audio_file_paths: list[str], show_progress: bool = True
    ) -> list[Sequence[float]]:
        return [self._values["audio"] for _ in audio_file_paths]

    async def aget_image_embedding_batch(
        self, img_file_paths: list[str], show_progress: bool = True
    ) -> list[Sequence[float]]:
        return [self._values["image"] for _ in img_file_paths]

    async def aget_video_embedding_batch(
        self, video_file_paths: list[str], show_progress: bool = True
    ) -> list[Sequence[float]]:
        return [self._values["video"] for _ in video_file_paths]


class FakeVectorStore:
    def __init__(
        self, nodes: Optional[List[BaseNode]] = None, sims: Optional[List[float]] = None
    ) -> None:
        default_node = TextNode(text="text", id_="node-1")
        self.nodes = nodes or [default_node]
        self.sims = sims or [0.42]
        self.queries: list[Any] = []

    async def aquery(self, query, **kwargs):
        self.queries.append((query, kwargs))
        return SimpleNamespace(nodes=self.nodes, similarities=self.sims)


class FakeDocStore:
    def __init__(self) -> None:
        self._nodes: Dict[str, BaseNode] = {
            "node-1": TextNode(
                text="overwritten", id_="node-1", metadata={"source": "docstore"}
            )
        }

    def document_exists(self, node_id: str) -> bool:
        return node_id in self._nodes

    def get_node(self, node_id: str) -> BaseNode:
        return self._nodes[node_id]


class DummyVectorStoreIndex:
    def __init__(
        self,
        *,
        embed_model: Optional[DummyEmbedModel] = None,
        vector_store: Optional[FakeVectorStore] = None,
        docstore: Optional[FakeDocStore] = None,
    ) -> None:
        self._embed_model = embed_model
        self.vector_store = vector_store or FakeVectorStore()
        self.docstore = docstore or FakeDocStore()


def setup_bedrock_mock(
    monkeypatch, *, mock_read: bool = True, mock_single: bool = True
):
    calls: list[dict[str, Any]] = []

    def fake_init(self, model_name="nova", **kwargs):
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "additional_kwargs", kwargs)
        object.__setattr__(self, "embed_batch_size", kwargs.get("embed_batch_size", 8))
        object.__setattr__(
            self,
            "callback_manager",
            SimpleNamespace(
                on_event_start=lambda *a, **k: "evt",
                on_event_end=lambda *a, **k: None,
            ),
        )

    def fake_invoke(self, body):
        calls.append(body)
        return [0.1, 0.2]

    def fake_read(self, media, expected_exts, fallback_format_key):
        return ("ZW1iZWQ=", "mp3")

    monkeypatch.setattr(
        "raggify.llama_like.embeddings.bedrock.BedrockEmbedding.__init__",
        fake_init,
        raising=False,
    )
    if mock_single:
        monkeypatch.setattr(
            "raggify.llama_like.embeddings.bedrock.MultiModalBedrockEmbedding._invoke_single_embedding",
            fake_invoke,
        )
    if mock_read:
        monkeypatch.setattr(
            "raggify.llama_like.embeddings.bedrock.MultiModalBedrockEmbedding._read_media_payload",
            fake_read,
        )
    return calls


def setup_clap_mock(monkeypatch):
    class FakeVector(list):
        def tolist(self):
            return list(self)

    class FakeCLAPModule:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def load_ckpt(self, model_id):
            self.model_id = model_id

        def get_text_embedding(self, x):
            return [FakeVector([0.1, 0.2]) for _ in x]

        def get_audio_embedding_from_filelist(self, x):
            return [FakeVector([0.3, 0.4]) for _ in x]

    fake_module = SimpleNamespace(CLAP_Module=FakeCLAPModule)
    monkeypatch.setitem(sys.modules, "laion_clap", fake_module)
    return fake_module
