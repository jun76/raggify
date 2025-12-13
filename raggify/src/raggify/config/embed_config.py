from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any

from mashumaro import DataClassDictMixin

__all__ = ["EmbedProvider", "EmbedModel", "EmbedConfig"]


class EmbedProvider(StrEnum):
    CLIP = auto()
    OPENAI = auto()
    COHERE = auto()
    HUGGINGFACE = auto()
    CLAP = auto()
    VOYAGE = auto()
    BEDROCK = auto()


class EmbedModel(StrEnum):
    NAME = auto()
    ALIAS = auto()
    DIM = auto()


@dataclass(kw_only=True)
class EmbedConfig(DataClassDictMixin):
    """Config dataclass for embedding settings."""

    batch_size: int = 1000
    batch_interval_sec: int = 1

    # Text
    openai_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "text-embedding-3-large",
            EmbedModel.ALIAS.value: "te3l",
            EmbedModel.DIM.value: 1536,
        }
    )
    cohere_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "embed-v4.0",
            EmbedModel.ALIAS.value: "emv4",
            EmbedModel.DIM.value: 1536,
        }
    )
    clip_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "ViT-B/32",
            EmbedModel.ALIAS.value: "vi32",
            EmbedModel.DIM.value: 512,
        }
    )
    clap_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "laion/clap-htsat-unfused",
            EmbedModel.ALIAS.value: "lchu",
            EmbedModel.DIM.value: 512,
        }
    )
    huggingface_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "intfloat/multilingual-e5-base",
            EmbedModel.ALIAS.value: "imeb",
            EmbedModel.DIM.value: 768,
        }
    )
    voyage_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "voyage-3.5",
            EmbedModel.ALIAS.value: "vo35",
            EmbedModel.DIM.value: 2048,
        }
    )
    bedrock_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "amazon.nova-2-multimodal-embeddings-v1:0",
            EmbedModel.ALIAS.value: "n2v1",
            EmbedModel.DIM.value: 1024,
        }
    )

    # Image
    cohere_embed_model_image: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "embed-v4.0",
            EmbedModel.ALIAS.value: "emv4",
            EmbedModel.DIM.value: 1536,
        }
    )
    clip_embed_model_image: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "ViT-B/32",
            EmbedModel.ALIAS.value: "vi32",
            EmbedModel.DIM.value: 512,
        }
    )
    huggingface_embed_model_image: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "llamaindex/vdr-2b-multi-v1",
            EmbedModel.ALIAS.value: "v2m1",
            EmbedModel.DIM.value: 1536,
        }
    )
    voyage_embed_model_image: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "voyage-multimodal-3",
            EmbedModel.ALIAS.value: "vom3",
            EmbedModel.DIM.value: 1024,
        }
    )
    bedrock_embed_model_image: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "amazon.nova-2-multimodal-embeddings-v1:0",
            EmbedModel.ALIAS.value: "n2v1",
            EmbedModel.DIM.value: 1024,
        }
    )

    # Audio
    clap_embed_model_audio: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "laion/clap-htsat-unfused",
            EmbedModel.ALIAS.value: "lchu",
            EmbedModel.DIM.value: 512,
        }
    )
    bedrock_embed_model_audio: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "amazon.nova-2-multimodal-embeddings-v1:0",
            EmbedModel.ALIAS.value: "n2v1",
            EmbedModel.DIM.value: 1024,
        }
    )

    # Video
    bedrock_embed_model_video: dict[str, Any] = field(
        default_factory=lambda: {
            EmbedModel.NAME.value: "amazon.nova-2-multimodal-embeddings-v1:0",
            EmbedModel.ALIAS.value: "n2v1",
            EmbedModel.DIM.value: 1024,
        }
    )
