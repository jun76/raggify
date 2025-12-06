from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto

__all__ = ["LLMProvider", "LLMConfig"]


class LLMProvider(StrEnum):
    OPENAI = auto()
    HUGGINGFACE = auto()


@dataclass(kw_only=True)
class LLMConfig:
    """Config dataclass for LLM settings."""

    # Text
    openai_text_summarize_transform_model: str = "gpt-4o-mini"
    huggingface_text_summarize_transform_model: str = "Qwen/Qwen2-VL-2B-Instruct"

    # Image
    openai_image_summarize_transform_model: str = "gpt-4o-mini"
    huggingface_image_summarize_transform_model: str = "Qwen/Qwen2-VL-2B-Instruct"

    # Audio
    # TODO: Add audio summarize transform model configs

    # Video
    # TODO: Add video summarize transform model configs
