from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .default_settings import DefaultSettings


@dataclass(kw_only=True)
class EmbedConfig:
    """埋め込み関連の設定用データクラス"""

    # Text
    openai_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: dict(DefaultSettings.OPENAI_EMBED_MODEL_TEXT)
    )
    cohere_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: dict(DefaultSettings.COHERE_EMBED_MODEL_TEXT)
    )
    clip_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: dict(DefaultSettings.CLIP_EMBED_MODEL_TEXT)
    )
    huggingface_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: dict(DefaultSettings.HUGGINGFACE_EMBED_MODEL_TEXT)
    )
    voyage_embed_model_text: dict[str, Any] = field(
        default_factory=lambda: dict(DefaultSettings.VOYAGE_EMBED_MODEL_TEXT)
    )

    # Image
    cohere_embed_model_image: dict[str, Any] = field(
        default_factory=lambda: dict(DefaultSettings.COHERE_EMBED_MODEL_IMAGE)
    )
    clip_embed_model_image: dict[str, Any] = field(
        default_factory=lambda: dict(DefaultSettings.CLIP_EMBED_MODEL_IMAGE)
    )
    huggingface_embed_model_image: dict[str, Any] = field(
        default_factory=lambda: dict(DefaultSettings.HUGGINGFACE_EMBED_MODEL_IMAGE)
    )
    voyage_embed_model_image: dict[str, Any] = field(
        default_factory=lambda: dict(DefaultSettings.VOYAGE_EMBED_MODEL_IMAGE)
    )

    # Audio
    clap_embed_model_audio: dict[str, Any] = field(
        default_factory=lambda: dict(DefaultSettings.CLAP_EMBED_MODEL_AUDIO)
    )
