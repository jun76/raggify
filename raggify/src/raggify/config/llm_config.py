from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto

__all__ = ["LLMProvider", "LLMConfig"]


class LLMProvider(StrEnum):
    OPENAI = auto()


@dataclass(kw_only=True)
class LLMConfig:
    """Config dataclass for LLM settings."""

    # Image
    openai_image_caption_transform_model: str = "gpt-4o-mini"

    # Audio
    openai_audio_caption_transform_model: str = "gpt-4o-audio-preview"

    # Video
    openai_video_caption_transform_model: str = "gpt-4o-audio-preview"
