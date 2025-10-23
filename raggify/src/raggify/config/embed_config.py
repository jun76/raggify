from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .default_settings import DefaultSettings


@dataclass(kw_only=True)
class EmbedConfig:
    """埋め込み関連の設定用データクラス"""

    # Text
    openai_embed_model_text: str = DefaultSettings.OPENAI_EMBED_MODEL_TEXT
    cohere_embed_model_text: str = DefaultSettings.COHERE_EMBED_MODEL_TEXT
    clip_embed_model_text: str = DefaultSettings.CLIP_EMBED_MODEL_TEXT
    huggingface_embed_model_text: str = DefaultSettings.HUGGINGFACE_EMBED_MODEL_TEXT
    voyage_embed_model_text: str = DefaultSettings.VOYAGE_EMBED_MODEL_TEXT

    # Image
    cohere_embed_model_image: str = DefaultSettings.COHERE_EMBED_MODEL_IMAGE
    clip_embed_model_image: str = DefaultSettings.CLIP_EMBED_MODEL_IMAGE
    huggingface_embed_model_image: str = DefaultSettings.HUGGINGFACE_EMBED_MODEL_IMAGE
    voyage_embed_model_image: str = DefaultSettings.VOYAGE_EMBED_MODEL_IMAGE

    # Audio
    clap_embed_model_audio: Literal[
        "effect_short", "effect_varlen", "music", "speech", "general"
    ] = DefaultSettings.CLAP_EMBED_MODEL_AUDIO
