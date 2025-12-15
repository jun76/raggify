from __future__ import annotations

from enum import StrEnum, auto

from llama_index.core.constants import DATA_KEY
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.storage.docstore import keyval_docstore
from llama_index.core.storage.docstore import utils as docstore_utils

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

    @classmethod
    def class_name(cls) -> str:
        """Return class name for serialization."""
        return "AudioNode"


class VideoNode(TextNode):
    """Node implementation for video modality."""

    def __init__(self, *args, **kwargs) -> None:
        """Constructor."""
        super().__init__(*args, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        """Return class name for serialization."""
        return "VideoNode"


def pipe_load_hook() -> None:
    """Patch docstore loader to support llama-like schema nodes."""
    original = docstore_utils.json_to_doc

    def _patched(doc_dict: dict) -> BaseNode:
        data = doc_dict.get(DATA_KEY)
        if not isinstance(data, dict):
            return original(doc_dict)

        cls_name = data.get("class_name")
        if cls_name == AudioNode.class_name():
            return AudioNode.from_dict(data)
        if cls_name == VideoNode.class_name():
            return VideoNode.from_dict(data)

        return original(doc_dict)

    docstore_utils.json_to_doc = _patched
    setattr(keyval_docstore, "json_to_doc", _patched)
