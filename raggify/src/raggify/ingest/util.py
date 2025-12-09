from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from ..logger import logger

__all__ = ["MediaConverter"]


class MediaConverter:
    def __init__(self) -> None:
        """Constractor.

        Raises:
            ImportError: If ffmpeg is not installed.
        """
        try:
            import ffmpeg  # type: ignore
        except ImportError:
            from ..core.const import EXTRA_PKG_NOT_FOUND_MSG

            raise ImportError(
                EXTRA_PKG_NOT_FOUND_MSG.format(
                    pkg="ffmpeg-python (additionally, ffmpeg itself must be installed separately)",
                    extra="audio",
                    feature="AudioReader",
                )
            )

        self._ffmpeg = ffmpeg

    def audio_to_mp3(
        self, src: str, path: Path, sample_rate: int = 16000, bitrate: str = "192k"
    ) -> Optional[Path]:
        """Convert audio file to mp3 format.

        Args:
            src (str): Source audio file path.
            path (Path): Destination mp3 file path.
            sample_rate (int, optional): Target sample rate. Defaults to 16000.
            bitrate (str, optional): Audio bitrate string. Defaults to "192k".

        Returns:
            Optional[Path]: Converted audio file path, or None on failure.
        """
        try:
            (
                self._ffmpeg.input(src)
                .output(
                    str(path),
                    acodec="libmp3lame",
                    audio_bitrate=bitrate,
                    format="mp3",
                    ar=sample_rate,
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logger.error(f"{src} ffmpeg audio conversion failure: {e}")
            return None

        return path

    def extract_mp3_audio_from_video(
        self, src: str, path: Path, sample_rate: int = 16000
    ) -> Optional[Path]:
        """Extract mp3 audio track from video file.

        Args:
            src (str): Source video file path.
            path (Path): Destination mp3 file path.
            sample_rate (int, optional): Target sample rate. Defaults to 16000.
        """
        try:
            (
                self._ffmpeg.input(src)
                .output(
                    path,
                    acodec="libmp3lame",
                    ac=1,
                    ar=sample_rate,
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logger.error(f"{src} ffmpeg audio extraction failure: {e}")
            return None

        return path

    def extract_png_frames_from_video(
        self, src: str, frame_rate: int, base_path: Path
    ) -> Optional[Path]:
        """Extract png frames from video file.

        Args:
            src (str): Source video file path.
            frame_rate (int): Frame extraction rate (frames per second).
            base_path (Path): Base path for extracted png frames.

        Returns:
            Optional[Path]: Directory path containing extracted png frames, or None on failure.
        """
        temp_dir = base_path.parent / f"{base_path.stem}_frames"

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        temp_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(temp_dir / f"{base_path.stem}_%05d.png")
        try:
            (
                self._ffmpeg.input(src)
                .output(pattern, vf=f"fps={frame_rate}")
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logger.error(f"{src} ffmpeg frame extraction failure: {e}")
            return None

        return temp_dir

    def split_video(
        self, src: str, chunk_seconds: int, base_path: Path
    ) -> Optional[Path]:
        """Split video file into chunks.

        Args:
            src (str): Source video file path.
            chunk_seconds (int): Chunk length in seconds.
            base_path (Path): Base path for output video chunks.

        Returns:
            Optional[Path]: Directory path containing video chunks, or None on failure.
        """
        temp_dir = base_path.parent / f"{base_path.stem}_chunks"

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        temp_dir.mkdir(parents=True, exist_ok=True)
        ext = base_path.suffix
        pattern = temp_dir / f"{base_path.stem}_%05d{ext}"
        try:
            (
                self._ffmpeg.input(src)
                .output(
                    str(pattern),
                    f="segment",
                    segment_time=str(chunk_seconds),
                    c="copy",
                    reset_timestamps="1",
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logger.error(f"{src} ffmpeg video splitting failure: {e}")
            return None

        return temp_dir
