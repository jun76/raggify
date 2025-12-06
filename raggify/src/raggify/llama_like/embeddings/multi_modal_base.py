from __future__ import annotations

from abc import abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, Union

from llama_index.core.embeddings import BaseEmbedding, MultiModalEmbedding

if TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import Embedding

__all__ = ["AudioEmbedding", "VideoEmbedding", "AudioType", "VideoType"]

AudioType = Union[str, BytesIO]
VideoType = Union[str, BytesIO]


class AudioEmbedding(BaseEmbedding):
    """Abstract audio embedding class.

    MultiModalEmbedding currently lacks audio support, so this class serves
    as an intermediate abstraction for audio embeddings.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Constructor."""
        super().__init__(*args, **kwargs)

    @abstractmethod
    async def aget_audio_embedding_batch(
        self, audio_file_paths: list[AudioType], show_progress: bool = False
    ) -> list[Embedding]:
        """Async batch interface for audio embeddings.

        Args:
            audio_file_paths (list[AudioType]): Audio file paths.
            show_progress (bool, optional): Whether to show progress. Defaults to False.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        ...


class VideoEmbedding(AudioEmbedding, MultiModalEmbedding):
    """Abstract video embedding class.

    MultiModalEmbedding currently lacks video support, so this class serves
    as an intermediate abstraction for video embeddings.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Constructor."""
        super().__init__(*args, **kwargs)

    @abstractmethod
    async def aget_video_embedding_batch(
        self, video_file_paths: list[VideoType], show_progress: bool = False
    ) -> list[Embedding]:
        """Async batch interface for video embeddings.

        Args:
            video_file_paths (list[VideoType]): Video file paths.
            show_progress (bool, optional): Whether to show progress. Defaults to False.

        Returns:
            list[Embedding]: Embedding vectors.
        """
        ...
