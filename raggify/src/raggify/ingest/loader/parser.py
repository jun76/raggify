from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ...core.event import async_loop_runner
from ...core.exts import Exts
from ...logger import logger
from .file_reader import AudioReader, DummyMediaReader, MultiPDFReader, VideoReader

if TYPE_CHECKING:
    from llama_index.core.readers.base import BaseReader
    from llama_index.core.schema import Document

    from ...config.general_config import GeneralConfig


__all__ = ["DefaultParser"]


class DefaultParser:
    """Default parser that reads local files and generates documents."""

    def __init__(self, cfg: GeneralConfig, ingest_target_exts: set[str]) -> None:
        """Constructor.

        Args:
            cfg (GeneralConfig): General configuration.
            ingest_target_exts (set[str]): Allowed extensions for ingestion.
        """
        self._ingest_target_exts = ingest_target_exts
        self._readers: dict[str, BaseReader] = {}

        if cfg.image_embed_provider is not None:
            # Dictionary of custom readers to pass to SimpleDirectoryReader
            self._readers: dict[str, BaseReader] = {Exts.PDF: MultiPDFReader()}

            if cfg.use_modality_fallback:
                # add readers for image transcription if supported in the future
                pass

        if cfg.audio_embed_provider is not None:
            # Convert audio files to mp3 for ingestion
            audio_reader = AudioReader()
            for ext in Exts.AUDIO:
                self._readers[ext] = audio_reader

            if cfg.use_modality_fallback:
                # add readers for audio transcription if supported in the future
                pass

        # For cases like video -> image + audio decomposition, use a reader
        if cfg.video_embed_provider is None:
            if cfg.use_modality_fallback:
                video_reader = VideoReader()
                for ext in Exts.VIDEO:
                    self._readers[ext] = video_reader

        # For other media types, use dummy reader to pass through
        dummy_reader = DummyMediaReader()
        for ext in Exts.PASS_THROUGH_MEDIA:
            self._readers.setdefault(ext, dummy_reader)

    async def aparse(
        self,
        root: str,
    ) -> list[Document]:
        """Parse data asynchronously from the input path.
        Args:
            root (str): Target path.

        Returns:
            list[Document]: List of documents parsed from the input path(s).
        """
        from llama_index.core.readers.file.base import SimpleDirectoryReader

        try:
            path = Path(root).absolute()
            if path.is_file():
                ext = Exts.get_ext(root)
                if ext not in self._ingest_target_exts:
                    logger.warning(f"skip unsupported extension: {ext}")
                    return []

            reader = SimpleDirectoryReader(
                input_dir=root if path.is_dir() else None,
                input_files=[root] if path.is_file() else None,
                recursive=True,
                required_exts=list(self._ingest_target_exts),
                file_extractor=self._readers,
                raise_on_error=True,
            )

            docs = await reader.aload_data(show_progress=True)
        except Exception as e:
            logger.exception(e)
            raise ValueError("failed to parse from path") from e

        return docs

    def parse(
        self,
        root: str,
    ) -> list[Document]:
        """Parse data from the input path.
        Args:
            root (str): Target path.

        Returns:
            list[Document]: List of documents parsed from the input path(s).
        """
        return async_loop_runner.run(lambda: self.aparse(root))
