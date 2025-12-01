from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ....core.exts import Exts
from ....core.metadata import BasicMetaData
from ....core.utils import get_temp_file_path_from
from ....logger import logger

__all__ = ["AudioReader"]


class AudioReader(BaseReader):
    """Reader that converts audio files to mp3 for downstream ingestion."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        bitrate: str = "192k",
        suffix: str = ".mp3",
    ) -> None:
        """Constructor.

        Args:
            sample_rate (int, optional): Target sample rate. Defaults to 16000.
            bitrate (str, optional): Audio bitrate string. Defaults to "192k".
            suffix (str, optional): Output file extension. Defaults to ".mp3".
        """
        super().__init__()
        self._sample_rate = sample_rate
        self._bitrate = bitrate
        self._suffix = suffix

    def _convert(self, src: str) -> Path | None:
        """Execute audio conversion.

        Args:
            src (str): Source audio file path.

        Raises:
            ImportError: If ffmpeg is not installed.

        Returns:
            Path | None: Converted audio file path, or None on failure.
        """
        try:
            import ffmpeg  # type: ignore
        except ImportError:
            from ....core.const import PKG_NOT_FOUND_MSG

            raise ImportError(
                PKG_NOT_FOUND_MSG.format(
                    pkg="ffmpeg-python (additionally, ffmpeg itself must be installed separately)",
                    extra="audio",
                    feature="AudioReader",
                )
            )

        temp_path = Path(get_temp_file_path_from(source=src, suffix=self._suffix))
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            (
                ffmpeg.input(src)
                .output(
                    str(temp_path),
                    acodec="libmp3lame",
                    audio_bitrate=self._bitrate,
                    ac=1,
                    ar=self._sample_rate,
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as exc:
            logger.warning(f"ffmpeg audio convert failure {src}: {exc}")
            return None

        return temp_path

    def lazy_load_data(self, path: str, extra_info: Any = None) -> Iterable[Document]:
        """Convert audio files and return document placeholders.

        Args:
            path (str): File path.

        Returns:
            Iterable[Document]: Documents referencing converted files.
        """
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            logger.warning(f"file not found: {abs_path}")
            return []

        if not Exts.endswith_exts(abs_path, Exts.AUDIO):
            logger.warning(
                f"unsupported audio ext: {abs_path}. supported: {' '.join(Exts.AUDIO)}"
            )
            return []

        converted = self._convert(abs_path)
        if converted is None:
            return []

        meta = BasicMetaData()
        meta.file_path = str(converted)
        meta.temp_file_path = str(converted)
        meta.base_source = abs_path

        logger.debug(f"converted audio {abs_path} -> {converted}")

        return [Document(text=abs_path, metadata=meta.to_dict())]
