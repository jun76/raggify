from __future__ import annotations

from types import SimpleNamespace

__all__ = ["patch_openai_llm", "DummyOpenAI"]


class DummyOpenAI:
    """Minimal OpenAI stub recording initialization kwargs."""

    instances: list["DummyOpenAI"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        DummyOpenAI.instances.append(self)


def patch_openai_llm(monkeypatch) -> DummyOpenAI:
    """Patch llama_index.llms.openai.OpenAI with DummyOpenAI."""

    DummyOpenAI.instances = []
    monkeypatch.setattr("llama_index.llms.openai.OpenAI", DummyOpenAI)
    return DummyOpenAI
