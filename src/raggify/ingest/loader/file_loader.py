from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from ...core.exts import Exts
from ...logger import logger
from ...runtime import get_runtime as _rt
from .loader import Loader
from .reader import DummyMediaReader, MultiPDFReader, VideoReader

if TYPE_CHECKING:
    from llama_index.core.readers.base import BaseReader
    from llama_index.core.schema import ImageNode, TextNode

    from ...llama.core.schema import AudioNode, VideoNode


class FileLoader(Loader):
    """Loader that reads local files and generates nodes."""

    def __init__(self, persist_dir: Optional[Path]) -> None:
        """Constructor.

        Args:
            persist_dir (Optional[Path]): Persist directory.
        """
        super().__init__(persist_dir)

        # Dictionary of custom readers to pass to SimpleDirectoryReader
        self._readers: dict[str, BaseReader] = {Exts.PDF: MultiPDFReader()}

        # For cases like video -> image + audio decomposition, use a reader
        cfg = _rt().cfg.general
        if cfg.use_modality_fallback:
            if cfg.video_embed_provider is None:
                video_reader = VideoReader()
                for ext in Exts.VIDEO:
                    self._readers[ext] = video_reader

            # TODO: Add readers for image/audio transcription if supported in the future

        dummy_reader = DummyMediaReader()
        for ext in Exts.PASS_THROUGH_MEDIA:
            self._readers[ext] = dummy_reader

    async def aload_from_path(
        self,
        root: str,
        is_canceled: Callable[[], bool],
    ) -> tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
        """Load content from a local path and generate nodes.

        Directories are traversed recursively to ingest multiple files.

        Args:
            root (str): Target path.
            is_canceled (Callable[[], bool]): Whether this job has been canceled.

        Raises:
            ValueError: For invalid path or load errors.

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
                Text, image, audio, and video nodes.
        """
        from llama_index.core.readers.file.base import SimpleDirectoryReader

        try:
            path = Path(root).absolute()
            allowed_exts = set(Exts.FETCH_TARGET) | set(
                _rt().cfg.ingest.additional_exts
            )
            if path.is_file():
                ext = Exts.get_ext(root)
                if ext not in allowed_exts:
                    logger.warning(f"skip unsupported extension: {ext}")
                    return [], [], [], []

            reader = SimpleDirectoryReader(
                input_dir=root if path.is_dir() else None,
                input_files=[root] if path.is_file() else None,
                recursive=True,
                required_exts=list(allowed_exts),
                file_extractor=self._readers,
                raise_on_error=True,
            )

            docs = await reader.aload_data(show_progress=True)
        except Exception as e:
            logger.exception(e)
            raise ValueError("failed to load from path") from e

        return await self._asplit_docs_modality(docs=docs, is_canceled=is_canceled)

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
            try:
                temp_text, temp_image, temp_audio, temp_video = (
                    await self.aload_from_path(root=path, is_canceled=is_canceled)
                )
                texts.extend(temp_text)
                images.extend(temp_image)
                audios.extend(temp_audio)
                videos.extend(temp_video)
            except Exception as e:
                logger.exception(e)
                continue

        return texts, images, audios, videos
