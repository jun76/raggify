from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Optional

from llama_index.core.base.llms.types import (
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms import LLM, ChatMessage
from pydantic import Field, PrivateAttr

from raggify.llm.llm_manager import LLMContainer, LLMManager, LLMUsage
from tests.utils.mock_embed import DummyAudioBase, DummyMultiModalBase, DummyVideoBase

__all__ = [
    "DummyLLM",
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


class DummyLLM(LLM):
    """Simple LLM stub that records inputs."""

    response_text: str = Field(default="summary")
    error: Exception | None = Field(default=None, exclude=True)
    _calls: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._calls

    def _record(self, entry: dict[str, Any]) -> None:
        self._calls.append(entry)

    def _maybe_raise(self) -> None:
        if self.error is not None:
            raise self.error

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(is_chat_model=True, model_name="dummy")

    def complete(
        self,
        prompt: str,
        *,
        formatted: bool = False,
        image_documents: Optional[list] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        self._record(
            {
                "prompt": prompt,
                "formatted": formatted,
                "image_documents": image_documents,
            }
        )
        self._maybe_raise()
        return CompletionResponse(text=self.response_text)

    async def acomplete(
        self,
        prompt: str,
        *,
        formatted: bool = False,
        image_documents: Optional[list] = None,
        **kwargs: Any,
    ) -> CompletionResponse:
        return self.complete(
            prompt=prompt,
            formatted=formatted,
            image_documents=image_documents,
            **kwargs,
        )

    def stream_complete(
        self, prompt: str, *, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        self._maybe_raise()
        yield self.complete(prompt, formatted=formatted, **kwargs)

    async def astream_complete(
        self, prompt: str, *, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        self._maybe_raise()
        yield self.complete(prompt, formatted=formatted, **kwargs)

    def chat(self, messages, **kwargs: Any) -> ChatResponse:
        self._record({"messages": messages})
        self._maybe_raise()
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=self.response_text,
            )
        )

    async def achat(self, messages, **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream_chat(self, messages, **kwargs: Any) -> ChatResponseGen:
        self._maybe_raise()
        yield self.chat(messages, **kwargs)

    async def astream_chat(self, messages, **kwargs: Any) -> ChatResponseAsyncGen:
        self._maybe_raise()
        yield self.chat(messages, **kwargs)


class DummyRuntime:
    def __init__(self, llm_manager: LLMManager) -> None:
        self.llm_manager = llm_manager


def _build_llm_manager(
    *,
    text_llm: Optional[DummyLLM] = None,
    image_llm: Optional[DummyLLM] = None,
    audio_llm: Optional[DummyLLM] = None,
    video_llm: Optional[DummyLLM] = None,
) -> LLMManager:
    conts: dict[LLMUsage, LLMContainer] = {}
    text = text_llm or DummyLLM(response_text="text-summary")
    image = image_llm or DummyLLM(response_text="image-summary")
    audio = audio_llm or DummyLLM(response_text="audio-summary")
    video = video_llm or DummyLLM(response_text="video-summary")

    conts[LLMUsage.TEXT_SUMMARIZER] = LLMContainer(provider_name="dummy", llm=text)
    conts[LLMUsage.IMAGE_SUMMARIZER] = LLMContainer(provider_name="dummy", llm=image)
    conts[LLMUsage.AUDIO_SUMMARIZER] = LLMContainer(provider_name="dummy", llm=audio)
    conts[LLMUsage.VIDEO_SUMMARIZER] = LLMContainer(provider_name="dummy", llm=video)

    return LLMManager(conts)


def make_dummy_runtime(
    *,
    text_llm: Optional[DummyLLM] = None,
    image_llm: Optional[DummyLLM] = None,
    audio_llm: Optional[DummyLLM] = None,
    video_llm: Optional[DummyLLM] = None,
) -> DummyRuntime:
    manager = _build_llm_manager(
        text_llm=text_llm,
        image_llm=image_llm,
        audio_llm=audio_llm,
        video_llm=video_llm,
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
