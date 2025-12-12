from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from raggify.config.config_manager import ConfigManager
from raggify.config.general_config import GeneralConfig
from raggify.config.llm_config import LLMConfig, LLMProvider
from raggify.llm.llm import create_llm_manager
from raggify.llm.llm_manager import LLMUsage
from tests.utils.mock_llm import patch_openai_llm


def _make_cfg(
    *,
    text_provider: Any = None,
    image_provider: Any = None,
    audio_provider: Any = None,
    video_provider: Any = None,
    openai_base_url: str | None = None,
    text_model: str = "text-model",
    image_model: str = "image-model",
    audio_model: str = "audio-model",
    video_model: str = "video-model",
) -> ConfigManager:
    general = GeneralConfig()
    general.text_summarize_transform_provider = text_provider
    general.image_summarize_transform_provider = image_provider
    general.audio_summarize_transform_provider = audio_provider
    general.video_summarize_transform_provider = video_provider
    if openai_base_url is not None:
        general.openai_base_url = openai_base_url

    llm_cfg = LLMConfig()
    llm_cfg.openai_text_summarize_transform_model = text_model
    llm_cfg.openai_image_summarize_transform_model = image_model
    llm_cfg.openai_audio_summarize_transform_model = audio_model
    llm_cfg.openai_video_summarize_transform_model = video_model

    return cast(ConfigManager, SimpleNamespace(general=general, llm=llm_cfg))


def test_create_llm_manager_with_openai_providers(monkeypatch):
    dummy = patch_openai_llm(monkeypatch)
    cfg = _make_cfg(
        text_provider=LLMProvider.OPENAI,
        image_provider=LLMProvider.OPENAI,
        audio_provider=LLMProvider.OPENAI,
        video_provider=LLMProvider.OPENAI,
        openai_base_url="https://api.example.com",
        text_model="text-1",
        image_model="image-2",
        audio_model="audio-3",
        video_model="video-4",
    )

    manager = create_llm_manager(cfg)

    assert manager.llm_usage == {
        LLMUsage.TEXT_SUMMARIZER,
        LLMUsage.IMAGE_SUMMARIZER,
        LLMUsage.AUDIO_SUMMARIZER,
        LLMUsage.VIDEO_SUMMARIZER,
    }
    instances = dummy.instances
    assert [inst.kwargs["model"] for inst in instances] == [
        "text-1",
        "image-2",
        "audio-3",
        "video-4",
    ]
    assert manager.text_summarizer is instances[0]
    assert manager.image_summarizer is instances[1]
    assert manager.audio_summarizer is instances[2]
    assert manager.video_summarizer is instances[3]
    assert instances[2].kwargs["modalities"] == ["text"]
    assert instances[3].kwargs["modalities"] == ["text"]


def test_create_llm_manager_without_providers(monkeypatch):
    cfg = _make_cfg()
    manager = create_llm_manager(cfg)
    assert manager.llm_usage == set()


def test_create_llm_manager_raises_on_unsupported_provider():
    cfg = _make_cfg(text_provider=cast(LLMProvider, "unsupported"))
    with pytest.raises(RuntimeError, match="invalid LLM settings"):
        create_llm_manager(cfg)
