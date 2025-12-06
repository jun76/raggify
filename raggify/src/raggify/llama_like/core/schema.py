from __future__ import annotations

from enum import StrEnum, auto

from llama_index.core.schema import TextNode

__all__ = ["Modality", "AudioNode", "VideoNode"]


# Modalities
# ! Changing the string will change the space key and require reingest !
class Modality(StrEnum):
    TEXT = auto()
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()


class AudioNode(TextNode):
    """Node implementation for audio modality."""

    def __init__(self, *args, **kwargs) -> None:
        """Constructor."""
        super().__init__(*args, **kwargs)


class VideoNode(TextNode):
    """Node implementation for video modality."""

    def __init__(self, *args, **kwargs) -> None:
        """Constructor."""
        super().__init__(*args, **kwargs)
