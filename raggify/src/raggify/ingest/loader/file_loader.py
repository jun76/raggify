from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ...logger import logger
from .base_loader import BaseLoader

if TYPE_CHECKING:
    from llama_index.core.schema import ImageNode, TextNode

    from ...llama_like.core.schema import AudioNode, VideoNode
    from ..parser import BaseParser

__all__ = ["FileLoader"]


class FileLoader(BaseLoader):
    """Loader for local files that generates nodes."""

    def __init__(self, parser: BaseParser) -> None:
        """Constructor.

        Args:
            parser (Parser): Parser instance.
        """
        self._parser = parser

    async def aload_from_path(
        self, root: str
    ) -> tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
        """Load content from a local path and generate nodes.

        Directories are traversed recursively to ingest multiple files.

        Args:
            root (str): Target path.

        Raises:
            ValueError: For invalid path or load errors.

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
                Text, image, audio, and video nodes.
        """
        docs = await self._parser.aparse(root)
        logger.debug(f"loaded {len(docs)} docs from {root}")

        return await self._asplit_docs_modality(docs)

    async def aload_from_paths(
        self,
        paths: list[str],
        is_canceled: Callable[[], bool],
    ) -> tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
        """Load content from multiple paths and generate nodes.

        Args:
            paths (list[str]): Path list.
            is_canceled (Callable[[], bool]): Whether this job has been canceled.

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
                Text, image, audio, and video nodes.
        """
        texts = []
        images = []
        audios = []
        videos = []
        for path in paths:
            if is_canceled():
                logger.info("Job is canceled, aborting batch processing")
                return [], [], [], []
            try:
                temp_text, temp_image, temp_audio, temp_video = (
                    await self.aload_from_path(path)
                )
                texts.extend(temp_text)
                images.extend(temp_image)
                audios.extend(temp_audio)
                videos.extend(temp_video)
            except Exception as e:
                logger.exception(e)
                continue

        return texts, images, audios, videos
