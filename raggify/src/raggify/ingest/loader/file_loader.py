from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from ...config.ingest_config import IngestConfig
from ...logger import logger
from .base_loader import BaseLoader

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode, ImageNode, TextNode

    from ...llama_like.core.schema import AudioNode, VideoNode
    from ..parser import BaseParser

__all__ = ["FileLoader"]


class FileLoader(BaseLoader):
    """Loader for local files that generates nodes."""

    def __init__(self, parser: BaseParser, cfg: IngestConfig) -> None:
        """Constructor.

        Args:
            parser (Parser): Parser instance.
            cfg (IngestConfig): Ingest configuration.
        """
        super().__init__(cfg)
        self._parser = parser

    async def aload_from_path(self, root: str, force: bool) -> tuple[
        list[BaseNode],
        list[TextNode],
        list[ImageNode],
        list[AudioNode],
        list[VideoNode],
    ]:
        """Load content from a local path and generate nodes.

        Directories are traversed recursively to ingest multiple files.

        Args:
            root (str): Target path.
            force (bool): Whether to force reingestion even if already present.

        Raises:
            ValueError: For invalid path or load errors.

        Returns:
            tuple[
                list[BaseNode],
                list[TextNode],
                list[ImageNode],
                list[AudioNode],
                list[VideoNode],
            ]: Text tree, text leaf, image, audio, and video nodes.
        """
        docs = await self._parser.aparse(root=root, force=force)
        logger.debug(f"loaded {len(docs)} docs from {root}")

        return await self._asplit_docs_modality(docs)

    async def aload_from_paths(
        self,
        paths: list[str],
        force: bool,
        is_canceled: Callable[[], bool],
    ) -> tuple[
        list[BaseNode],
        list[TextNode],
        list[ImageNode],
        list[AudioNode],
        list[VideoNode],
    ]:
        """Load content from multiple paths and generate nodes.

        Args:
            paths (list[str]): Path list.
            force (bool): Whether to force reingestion even if already present.
            is_canceled (Callable[[], bool]): Whether this job has been canceled.

        Returns:
            tuple[
                list[BaseNode],
                list[TextNode],
                list[ImageNode],
                list[AudioNode],
                list[VideoNode],
            ]: Text tree, text leaf, image, audio, and video nodes.
        """
        text_trees = []
        text_leaves = []
        images = []
        audios = []
        videos = []
        for path in paths:
            if is_canceled():
                logger.info("Job is canceled, aborting batch processing")
                return [], [], [], [], []
            try:
                temp_text_tree, temp_text_leaf, temp_image, temp_audio, temp_video = (
                    await self.aload_from_path(root=path, force=force)
                )
                text_trees.extend(temp_text_tree)
                text_leaves.extend(temp_text_leaf)
                images.extend(temp_image)
                audios.extend(temp_audio)
                videos.extend(temp_video)
            except Exception as e:
                logger.exception(e)
                continue

        return text_trees, text_leaves, images, audios, videos
