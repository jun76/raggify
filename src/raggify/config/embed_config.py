from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any


class EmbedProvider(StrEnum):
    CLIP = auto()
    OPENAI = auto()
    COHERE = auto()
    HUGGINGFACE = auto()
    CLAP = auto()
    VOYAGE = auto()


class EmbedModel(StrEnum):
    NAME = auto()
    DIM = auto()


@dataclass(kw_only=True)
class EmbedConfig:
    """埋め込み関連の設定用データクラス"""

    # Text
    openai_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "text-embedding-3-small",
            EmbedModel.DIM.value: 1536,
        }
    )
    cohere_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "embed-v4.0",
            EmbedModel.DIM.value: 1536,
        }
    )
    clip_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "ViT-B/32",
            EmbedModel.DIM.value: 512,
        }
    )
    huggingface_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "intfloat/multilingual-e5-base",
            EmbedModel.DIM.value: 768,
        }
    )
    voyage_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "voyage-3.5",
            EmbedModel.DIM.value: 2048,
        }
    )

    # Image
    cohere_embed_model_image: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "embed-v4.0",
            EmbedModel.DIM.value: 1536,
        }
    )
    clip_embed_model_image: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "ViT-B/32",
            EmbedModel.DIM.value: 512,
        }
    )
    huggingface_embed_model_image: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "llamaindex/vdr-2b-multi-v1",
            EmbedModel.DIM.value: 1536,
        }
    )
    voyage_embed_model_image: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "voyage-multimodal-3",
            EmbedModel.DIM.value: 1024,
        }
    )

    # Audio
    clap_embed_model_audio: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "effect_varlen",
            EmbedModel.DIM.value: 512,
        }
    )
