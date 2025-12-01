from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ....core.const import PKG_NOT_FOUND_MSG
from ....core.exts import Exts
from ....core.metadata import BasicMetaData
from ....core.utils import get_temp_file_path_from
from ....logger import logger

__all__ = ["VideoReader"]


class VideoReader(BaseReader):
    """Reader that splits video files into frame images and audio tracks."""

    def __init__(
        self,
        *,
        fps: int = 1,
        audio_sample_rate: int = 16000,
        image_suffix: str = Exts.PNG,
        audio_suffix: str = Exts.WAV,
    ) -> None:
        """Constructor.

        Args:
            fps (int, optional): Frames per second to extract. Defaults to 1.
            audio_sample_rate (int, optional): Sample rate for audio extraction. Defaults to 16000.
            image_suffix (str, optional): Frame image extension. Defaults to Exts.PNG.
            audio_suffix (str, optional): Audio file extension. Defaults to Exts.WAV.

        Raises:
            ImportError: If ffmpeg is not installed.
            ValueError: If fps is zero or negative.
        """
        super().__init__()

        self._fps = fps
        self._audio_sample_rate = audio_sample_rate
        self._image_suffix = image_suffix
        self._audio_suffix = audio_suffix

    def _extract_frames(self, src: str) -> list[Path]:
        """Extract frame images from a video.

        Args:
            src (str): Video file path.

        Raises:
            ImportError: If ffmpeg is not installed.

        Returns:
            list[Path]: Extracted frame paths.
        """
        try:
            import ffmpeg  # type: ignore
        except ImportError:
            raise ImportError(
                PKG_NOT_FOUND_MSG.format(
                    pkg="ffmpeg-python (additionally, ffmpeg itself must be installed separately)",
                    extra="ffmpeg",
                    feature="ffmpeg",
                )
            )

        base_path = Path(get_temp_file_path_from(source=src, suffix=self._image_suffix))
        temp_dir = base_path.parent / f"{base_path.stem}_frames"

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        temp_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(temp_dir / f"{base_path.stem}_%05d{self._image_suffix}")
        try:
            (
                ffmpeg.input(src)
                .filter("fps", self._fps)
                .output(pattern, format="image2", vcodec="png")
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logger.warning(f"ffmpeg frame extraction from {src} failure: {e}")
            return []

        frames = sorted(temp_dir.glob(f"{base_path.stem}_*{self._image_suffix}"))
        logger.debug(f"extracted {len(frames)} frame(s) from {src}")

        return frames

    def _extract_audio(self, src: str) -> Path | None:
        """Extract an audio track from a video.

        Args:
            src (str): Video file path.

        Raises:
            ImportError: If ffmpeg is not installed.

        Returns:
            Path | None: Extracted audio file path.
        """
        try:
            import ffmpeg  # type: ignore
        except ImportError:
            raise ImportError(
                PKG_NOT_FOUND_MSG.format(
                    pkg="ffmpeg-python (additionally, ffmpeg itself must be installed separately)",
                    extra="ffmpeg",
                    feature="ffmpeg",
                )
            )

        temp_path = Path(get_temp_file_path_from(source=src, suffix=self._audio_suffix))
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        if temp_path.exists():
            temp_path.unlink()

        try:
            (
                ffmpeg.input(src)
                .output(
                    str(temp_path), acodec="pcm_s16le", ac=1, ar=self._audio_sample_rate
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except Exception as e:
            logger.warning(f"ffmpeg audio extraction from {src} failure: {e}")
            return None

        logger.debug(f"extracted 1 audio track from {src}")

        return temp_path

    def _image_docs(self, frame_paths: Sequence[Path], source: str) -> list[Document]:
        """Convert frame images to Document objects.

        Args:
            frame_paths (Sequence[Path]): Frame image paths.
            source (str): Source video path.

        Returns:
            list[Document]: Generated documents.
        """
        docs: list[Document] = []
        for i, frame_path in enumerate(frame_paths):
            meta = BasicMetaData()
            meta.file_path = str(frame_path)
            meta.temp_file_path = str(frame_path)
            meta.base_source = source
            meta.page_no = i

            docs.append(Document(text=source, metadata=meta.to_dict()))

        return docs

    def _audio_doc(self, audio_path: Path, source: str) -> Document:
        """Convert an audio file to a Document.

        Args:
            audio_path (Path): Audio file path.
            source (str): Source video path.

        Returns:
            Document: Generated audio document.
        """
        meta = BasicMetaData()
        meta.file_path = str(audio_path)
        meta.temp_file_path = str(audio_path)
        meta.base_source = source

        return Document(text=source, metadata=meta.to_dict())

    def _load_video(self, path: str, allowed_exts: Iterable[str]) -> list[Document]:
        """Load a video and generate frame and audio documents.

        Args:
            path (str): Video file path.
            allowed_exts (Iterable[str]): Allowed extensions.

        Raises:
            ValueError: If an unsupported extension is specified.

        Returns:
            list[Document]: Generated documents.
        """
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            logger.warning(f"file not found: {abs_path}")
            return []

        if not Exts.endswith_exts(abs_path, set(allowed_exts)):
            raise ValueError(
                f"unsupported video ext: {abs_path}. supported: {' '.join(allowed_exts)}"
            )

        frames = self._extract_frames(abs_path)
        audio = self._extract_audio(abs_path)
        docs = self._image_docs(frames, abs_path)
        if audio is not None:
            docs.append(self._audio_doc(audio, abs_path))
            logger.debug(
                f"loaded {len(frames)} image docs + 1 audio doc from {abs_path}"
            )
        else:
            logger.debug(
                f"loaded {len(frames)} image docs from {abs_path} (audio missing)"
            )

        return docs

    def lazy_load_data(self, path: str, extra_info: Any = None) -> Iterable[Document]:
        """Split a video file into image and audio documents.

        Args:
            path (str): Video file path.
            extra_info (Any, optional): Unused extra info. Defaults to None.

        Raises:
            ValueError: If the extension is unsupported.

        Returns:
            Iterable[Document]: Extracted documents.
        """
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            logger.warning(f"file not found: {abs_path}")
            return []

        return self._load_video(abs_path, allowed_exts=Exts.VIDEO)
