from __future__ import annotations

from enum import StrEnum, auto
from typing import Any

from llama_index.core.schema import TextNode


# Modalities
# ! Changing the string will change the space key and require reingest !
class Modality(StrEnum):
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()


class AudioNode(TextNode):
    """Node implementation for audio modality."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Constructor."""
        super().__init__(*args, **kwargs)


class VideoNode(TextNode):
    """Node implementation for video modality."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Constructor."""
        super().__init__(*args, **kwargs)
