from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from raggify.config.config_manager import ConfigManager
from raggify.config.embed_config import EmbedConfig, EmbedProvider
from raggify.config.general_config import GeneralConfig
from raggify.embed.embed import create_embed_manager
from raggify.llama_like.core.schema import Modality

from .config import configure_test_env

configure_test_env()


@pytest.fixture(autouse=True)
def mock_embedding_classes(monkeypatch):
    class DummyEmbedding:
        def __init__(self, *args, **kwargs) -> None:
            self.kwargs = kwargs

    targets = [
        "llama_index.embeddings.openai.base.OpenAIEmbedding",
        "llama_index.embeddings.cohere.base.CohereEmbedding",
        # "llama_index.embeddings.clip.ClipEmbedding",
        # "llama_index.embeddings.huggingface.HuggingFaceEmbedding",
        # "raggify.llama_like.embeddings.clap.ClapEmbedding",
        "llama_index.embeddings.voyageai.base.VoyageEmbedding",
        "raggify.llama_like.embeddings.bedrock.MultiModalBedrockEmbedding",
    ]

    for path in targets:
        monkeypatch.setattr(path, DummyEmbedding)


def _make_cfg(*, text=None, image=None, audio=None, video=None) -> ConfigManager:
    general = GeneralConfig(
        text_embed_provider=text,
        image_embed_provider=image,
        audio_embed_provider=audio,
        video_embed_provider=video,
        rerank_provider=None,
    )
    embed = EmbedConfig()
    stub = SimpleNamespace(general=general, embed=embed)
    return cast(ConfigManager, stub)


def _assert_provider(manager, modality: Modality, provider: EmbedProvider):
    container = manager.get_container(modality)
    assert container.provider_name == provider


@pytest.mark.parametrize(
    "provider",
    [
        EmbedProvider.OPENAI,
        EmbedProvider.COHERE,
        # EmbedProvider.CLIP,
        # EmbedProvider.HUGGINGFACE,
        EmbedProvider.VOYAGE,
        EmbedProvider.BEDROCK,
    ],
)
def test_create_embed_manager_text_variants(provider):
    cfg = _make_cfg(text=provider)
    manager = create_embed_manager(cfg)
    assert manager.modality == {Modality.TEXT}
    _assert_provider(manager, Modality.TEXT, provider)


@pytest.mark.parametrize(
    "provider",
    [
        EmbedProvider.COHERE,
        # EmbedProvider.CLIP,
        # EmbedProvider.HUGGINGFACE,
        EmbedProvider.VOYAGE,
        EmbedProvider.BEDROCK,
    ],
)
def test_create_embed_manager_image_variants(provider):
    cfg = _make_cfg(image=provider)
    manager = create_embed_manager(cfg)
    assert manager.modality == {Modality.IMAGE}
    _assert_provider(manager, Modality.IMAGE, provider)


@pytest.mark.parametrize(
    "provider",
    [
        # EmbedProvider.CLAP,
        EmbedProvider.BEDROCK,
    ],
)
def test_create_embed_manager_audio_variants(provider):
    cfg = _make_cfg(audio=provider)
    manager = create_embed_manager(cfg)
    assert manager.modality == {Modality.AUDIO}
    _assert_provider(manager, Modality.AUDIO, provider)


def test_create_embed_manager_video_variant():
    cfg = _make_cfg(video=EmbedProvider.BEDROCK)
    manager = create_embed_manager(cfg)
    assert manager.modality == {Modality.VIDEO}
    _assert_provider(manager, Modality.VIDEO, EmbedProvider.BEDROCK)


def test_create_embed_manager_requires_any_provider():
    cfg = _make_cfg()
    with pytest.raises(RuntimeError):
        create_embed_manager(cfg)
