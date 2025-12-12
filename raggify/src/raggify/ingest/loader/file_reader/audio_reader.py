from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ....logger import logger

__all__ = ["AudioReader"]


class AudioReader(BaseReader):
    """Reader that converts audio files to mp3 for downstream ingestion."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        bitrate: str = "192k",
    ) -> None:
        """Constructor.

        Args:
            sample_rate (int, optional): Target sample rate. Defaults to 16000.
            bitrate (str, optional): Audio bitrate string. Defaults to "192k".
        """
        super().__init__()
        self._sample_rate = sample_rate
        self._bitrate = bitrate

    def lazy_load_data(self, path: Any, extra_info: Any = None) -> Iterable[Document]:
        """Convert audio files and return document placeholders.

        Args:
            path (Any): File path-like object.

        Returns:
            Iterable[Document]: Documents referencing converted files.
        """
        from ....core.exts import Exts
        from ....core.metadata import BasicMetaData
        from ....core.utils import get_temp_path

        path = os.fspath(path)
        if not Exts.endswith_exts(path, Exts.AUDIO):
            logger.error(
                f"unsupported audio ext: {path}. supported: {' '.join(Exts.AUDIO)}"
            )
            return []

        if Exts.endswith_ext(path, Exts.MP3):
            meta = BasicMetaData()
            meta.file_path = path
            meta.base_source = path

            logger.debug(f"audio file is already mp3, skipping conversion: {path}")

            return [Document(text=path, metadata=meta.to_dict())]

        try:
            from ...util import MediaConverter

            converted = MediaConverter().audio_to_mp3(
                src=Path(path),
                dst=get_temp_path(seed=path, suffix=Exts.MP3),
                sample_rate=self._sample_rate,
                bitrate=self._bitrate,
            )
        except ImportError as e:
            logger.error(f"ffmpeg not installed, cannot read audio files: {e}")
            return []

        if converted is None:
            return []

        meta = BasicMetaData()
        meta.file_path = str(converted)
        meta.temp_file_path = str(converted)
        meta.base_source = path

        logger.debug(f"converted audio {path} -> {converted}")

        return [Document(text=path, metadata=meta.to_dict())]
