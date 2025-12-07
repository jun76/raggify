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
    import torch

    class DummyBatch(dict):
        def to(self, device):
            return DummyBatch({key: value.to(device) for key, value in self.items()})

    class FakeProcessor:
        def __init__(self) -> None:
            self.feature_extractor = SimpleNamespace(sampling_rate=48000)

        def __call__(self, *, text=None, audios=None, **kwargs):
            if text is not None:
                batch = len(text)
            else:
                batch = len(audios or [])
            tensor = torch.ones((batch, 2), dtype=torch.float32)
            return DummyBatch({"inputs": tensor})

    def _batch_size(kwargs):
        if not kwargs:
            return 1
        first = next(iter(kwargs.values()))
        return first.shape[0]

    class FakeClapModel:
        instances: list[FakeClapModel] = []

        def __init__(self) -> None:
            FakeClapModel.instances.append(self)
            self.device = "cpu"

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def get_text_features(self, **kwargs):
            batch = _batch_size(kwargs)
            return torch.tensor([[1.0, 0.0]] * batch, dtype=torch.float32)

        def get_audio_features(self, **kwargs):
            batch = _batch_size(kwargs)
            return torch.tensor([[0.0, 1.0]] * batch, dtype=torch.float32)

    processor_calls: list[tuple[tuple, dict]] = []
    model_calls: list[tuple[tuple, dict]] = []

    def fake_processor_from_pretrained(*args, **kwargs):
        processor_calls.append((args, kwargs))
        return FakeProcessor()

    def fake_model_from_pretrained(*args, **kwargs):
        model_calls.append((args, kwargs))
        return FakeClapModel()

    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr(
        "transformers.ClapModel.from_pretrained",
        fake_model_from_pretrained,
    )

    return SimpleNamespace(
        model_instances=FakeClapModel.instances,
        processor_calls=processor_calls,
        model_calls=model_calls,
    )
