from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional

from llama_index.core.llms import ChatMessage
from llama_index.core.base.llms.types import ChatResponse

from tests.utils.mock_embed import DummyAudioBase, DummyMultiModalBase, DummyVideoBase

__all__ = [
    "DummyLLM",
    "DummyLLMManager",
    "DummyWhisperModel",
    "apply_patch_embedding_bases",
    "make_dummy_runtime",
    "patch_dummy_whisper",
]


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

    def __init__(
        self, response_text: str = "summary", *, error: Exception | None = None
    ):
        self.response_text = response_text
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, prompt: str, image_documents: Optional[list] = None):
        self.calls.append({"prompt": prompt, "image_documents": image_documents})
        if self.error is not None:
            raise self.error
        return DummyLLMResponse(self.response_text)

    async def acomplete(self, *, prompt: str, image_documents: Optional[list] = None):
        return self.complete(prompt=prompt, image_documents=image_documents)

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages})
        if self.error is not None:
            raise self.error
        return ChatResponse(
            message=ChatMessage(role="assistant", content=self.response_text)
        )

    async def achat(self, messages, **kwargs):
        return self.chat(messages, **kwargs)


class DummyLLMManager:
    def __init__(
        self,
        *,
        text_summarize_transform: Optional[DummyLLM] = None,
        image_summarize_transform: Optional[DummyLLM] = None,
        audio_summarize_transform: Optional[DummyLLM] = None,
        video_summarize_transform: Optional[DummyLLM] = None,
    ) -> None:
        self.text_summarize_transform = text_summarize_transform or DummyLLM(
            "text-summary"
        )
        self.image_summarize_transform = image_summarize_transform or DummyLLM(
            "image-summary"
        )
        self.audio_summarize_transform = audio_summarize_transform
        self.video_summarize_transform = video_summarize_transform


class DummyRuntime:
    def __init__(self, llm_manager: DummyLLMManager) -> None:
        self.llm_manager = llm_manager


def make_dummy_runtime(
    *,
    text_llm: Optional[DummyLLM] = None,
    image_llm: Optional[DummyLLM] = None,
    audio_llm: Optional[DummyLLM] = None,
    video_llm: Optional[DummyLLM] = None,
) -> DummyRuntime:
    manager = DummyLLMManager(
        text_summarize_transform=text_llm,
        image_summarize_transform=image_llm,
        audio_summarize_transform=audio_llm,
        video_summarize_transform=video_llm,
    )
    return DummyRuntime(manager)


@dataclass
class DummyWhisperModel:
    transcript: str
    expected_path: Optional[str] = None
    calls: list[str] = field(default_factory=list)

    def transcribe(self, path: str) -> dict[str, str]:
        if self.expected_path is not None:
            assert path == self.expected_path

        self.calls.append(path)
        return {"text": self.transcript}


def patch_dummy_whisper(
    monkeypatch: Any, transcript: str, expected_path: Optional[str] = None
) -> DummyWhisperModel:
    model = DummyWhisperModel(transcript=transcript, expected_path=expected_path)
    dummy_module = SimpleNamespace(load_model=lambda *_args, **_kwargs: model)
    monkeypatch.setitem(sys.modules, "whisper", dummy_module)

    return model
