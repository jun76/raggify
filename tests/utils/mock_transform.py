from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from tests.utils.mock_embed import (
    DummyAudioBase,
    DummyMultiModalBase,
    DummyVideoBase,
)


def apply_patch_embedding_bases(monkeypatch: Any) -> None:
    """Patch embedding base classes used by transforms for tests."""

    monkeypatch.setattr(
        "llama_index.core.embeddings.multi_modal_base.MultiModalEmbedding",
        DummyMultiModalBase,
    )
    monkeypatch.setattr(
        "raggify.llama_like.embeddings.multi_modal_base.AudioEmbedding",
        DummyAudioBase,
    )
    monkeypatch.setattr(
        "raggify.llama_like.embeddings.multi_modal_base.VideoEmbedding",
        DummyVideoBase,
    )


@dataclass
class DummyLLMResponse:
    text: str


class DummyLLM:
    """Simple LLM stub that records inputs."""

    def __init__(self, response_text: str = "summary", *, error: Exception | None = None):
        self.response_text = response_text
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, prompt: str, image_documents: Optional[list] = None):
        self.calls.append({"prompt": prompt, "image_documents": image_documents})
        if self.error is not None:
            raise self.error
        return DummyLLMResponse(self.response_text)


class DummyLLMManager:
    def __init__(
        self,
        *,
        text_summarizer: Optional[DummyLLM] = None,
        image_summarizer: Optional[DummyLLM] = None,
    ) -> None:
        self.text_summarizer = text_summarizer or DummyLLM("text-summary")
        self.image_summarizer = image_summarizer or DummyLLM("image-summary")
        self.audio_summarizer = None
        self.video_summarizer = None


class DummyRuntime:
    def __init__(self, llm_manager: DummyLLMManager) -> None:
        self.llm_manager = llm_manager


def make_dummy_runtime(
    *,
    text_llm: Optional[DummyLLM] = None,
    image_llm: Optional[DummyLLM] = None,
) -> DummyRuntime:
    manager = DummyLLMManager(text_summarizer=text_llm, image_summarizer=image_llm)
    return DummyRuntime(manager)
