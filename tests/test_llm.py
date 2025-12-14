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
    image_provider: Any = None,
    audio_provider: Any = None,
    video_provider: Any = None,
    openai_base_url: str | None = None,
    image_model: str = "image-model",
    audio_model: str = "audio-model",
    video_model: str = "video-model",
) -> ConfigManager:
    general = GeneralConfig()
    general.image_caption_transform_provider = image_provider
    general.audio_caption_transform_provider = audio_provider
    general.video_caption_transform_provider = video_provider
    if openai_base_url is not None:
        general.openai_base_url = openai_base_url

    llm_cfg = LLMConfig()
    llm_cfg.openai_image_caption_transform_model = image_model
    llm_cfg.openai_audio_caption_transform_model = audio_model
    llm_cfg.openai_video_caption_transform_model = video_model

    return cast(ConfigManager, SimpleNamespace(general=general, llm=llm_cfg))


def test_create_llm_manager_with_openai_providers(monkeypatch):
    dummy = patch_openai_llm(monkeypatch)
    cfg = _make_cfg(
        image_provider=LLMProvider.OPENAI,
        audio_provider=LLMProvider.OPENAI,
        video_provider=LLMProvider.OPENAI,
        openai_base_url="https://api.example.com",
        image_model="image-2",
        audio_model="audio-3",
        video_model="video-4",
    )

    manager = create_llm_manager(cfg)

    assert manager.llm_usage == {
        LLMUsage.IMAGE_CAPTIONER,
        LLMUsage.AUDIO_CAPTIONER,
        LLMUsage.VIDEO_CAPTIONER,
    }
    instances = dummy.instances
    assert [inst.kwargs["model"] for inst in instances] == [
        "image-2",
        "audio-3",
        "video-4",
    ]
    assert manager.image_captioner is instances[0]
    assert manager.audio_captioner is instances[1]
    assert manager.video_captioner is instances[2]
    assert instances[1].kwargs["modalities"] == ["text"]
    assert instances[2].kwargs["modalities"] == ["text"]


def test_create_llm_manager_without_providers(monkeypatch):
    cfg = _make_cfg()
    manager = create_llm_manager(cfg)
    assert manager.llm_usage == set()


def test_create_llm_manager_raises_on_unsupported_provider():
    cfg = _make_cfg(image_provider=cast(LLMProvider, "unsupported"))
    with pytest.raises(RuntimeError, match="invalid LLM settings"):
        create_llm_manager(cfg)
