from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ....core.exts import Exts
from ....core.metadata import BasicMetaData
from ....core.utils import get_temp_path
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

    def _convert(self, src: Path) -> Optional[Path]:
        """Execute audio conversion.

        Args:
            src (Path): Source audio file path.

        Raises:
            ImportError: If ffmpeg is not installed.

        Returns:
            Optional[Path]: Converted audio file path, or None on failure.
        """
        from ....core.exts import Exts
        from ...util import MediaConverter

        temp_path = get_temp_path(seed=str(src), suffix=Exts.MP3)
        converter = MediaConverter()

        return converter.audio_to_mp3(
            src=src,
            dst=temp_path,
            sample_rate=self._sample_rate,
            bitrate=self._bitrate,
        )

    def lazy_load_data(self, path: str, extra_info: Any = None) -> Iterable[Document]:
        """Convert audio files and return document placeholders.

        Args:
            path (str): File path.

        Returns:
            Iterable[Document]: Documents referencing converted files.
        """
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            logger.error(f"file not found: {abs_path}")
            return []

        if not Exts.endswith_exts(abs_path, Exts.AUDIO):
            logger.error(
                f"unsupported audio ext: {abs_path}. supported: {' '.join(Exts.AUDIO)}"
            )
            return []

        try:
            converted = self._convert(Path(abs_path))
        except ImportError as e:
            logger.error(f"ffmpeg not installed, cannot read audio files: {e}")
            return []

        if converted is None:
            return []

        meta = BasicMetaData()
        meta.file_path = str(converted)
        meta.temp_file_path = str(converted)
        meta.base_source = abs_path

        logger.debug(f"converted audio {abs_path} -> {converted}")

        return [Document(text=abs_path, metadata=meta.to_dict())]
