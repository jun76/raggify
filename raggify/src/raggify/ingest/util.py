from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..logger import logger

__all__ = ["MediaConverter"]


class MediaConverter:
    """Utility class for audio or video conversion using ffmpeg."""

    def __init__(self) -> None:
        """Constructor.

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
        self, src: Path, dst: Path, sample_rate: int = 16000, bitrate: str = "192k"
    ) -> Optional[Path]:
        """Convert audio file to mp3 format.

        Args:
            src (Path): Source audio file path.
            dst (Path): Destination mp3 file path.
            sample_rate (int, optional): Target sample rate. Defaults to 16000.
            bitrate (str, optional): Audio bitrate string. Defaults to "192k".

        Returns:
            Optional[Path]: Converted audio file path, or None on failure.
        """
        try:
            (
                self._ffmpeg.input(str(src))
                .output(
                    str(dst),
                    acodec="libmp3lame",
                    audio_bitrate=bitrate,
                    format="mp3",
                    ar=sample_rate,
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logger.error(f"failed to convert audio to mp3 for {src}: {e}")
            return None

        return dst

    def extract_mp3_audio_from_video(
        self, src: Path, dst: Path, sample_rate: int = 16000
    ) -> Optional[Path]:
        """Extract mp3 audio track from video file.

        Args:
            src (Path): Source video file path.
            dst (Path): Destination mp3 file path.
            sample_rate (int, optional): Target sample rate. Defaults to 16000.
        """
        try:
            (
                self._ffmpeg.input(str(src))
                .output(
                    str(dst),
                    acodec="libmp3lame",
                    ac=1,
                    ar=sample_rate,
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logger.error(f"failed to extract mp3 audio from video for {src}: {e}")
            return None

        return dst

    def extract_png_frames_from_video(
        self, src: Path, dst: Path, frame_rate: int
    ) -> Optional[Path]:
        """Extract png frames from video file.

        Args:
            src (Path): Source video file path.
            dst (Path): Directory path for extracted png frames.
            frame_rate (int): Frame extraction rate (frames per second).

        Returns:
            Optional[Path]: Directory path containing extracted png frames, or None on failure.
        """
        pattern = str(dst / "%05d.png")
        try:
            (
                self._ffmpeg.input(str(src))
                .output(pattern, vf=f"fps={frame_rate}")
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logger.error(f"failed to extract png frames from video for {src}: {e}")
            return None

        return dst

    def split(self, src: Path, dst: Path, chunk_seconds: int) -> Optional[Path]:
        """Split audio or video file into chunks.

        Args:
            src (Path): Source file path.
            dst (Path): Directory path for output chunks.
            chunk_seconds (int): Chunk length in seconds.

        Returns:
            Optional[Path]: Directory path containing chunks, or None on failure.
        """
        try:
            probe = self._ffmpeg.probe(src)
            duration = float(probe["format"]["duration"])
        except Exception as e:
            logger.error(f"failed to probe media duration for {src}: {e}")
            return None

        if duration is None or duration <= chunk_seconds:
            return None

        pattern = dst / f"%05d{dst.suffix}"
        try:
            (
                self._ffmpeg.input(str(src))
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
            logger.error(f"failed to split video for {src}: {e}")
            return None

        return dst
