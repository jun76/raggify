from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ....core.metadata import BasicMetaData
from ....logger import logger

__all__ = ["VideoReader"]


class VideoReader(BaseReader):
    """Reader that splits video files into frame images and audio tracks."""

    def __init__(
        self,
        *,
        fps: int = 1,
        audio_sample_rate: int = 16000,
    ) -> None:
        """Constructor.

        Args:
            fps (int, optional): Frames per second to extract. Defaults to 1.
            audio_sample_rate (int, optional): Sample rate for audio extraction. Defaults to 16000.
        """
        super().__init__()

        self._fps = fps
        self._audio_sample_rate = audio_sample_rate

    def _extract_frames(self, src: Path) -> list[Path]:
        """Extract frame images from a video.

        Args:
            src (Path): Video file path.

        Raises:
            ImportError: If ffmpeg is not installed.

        Returns:
            list[Path]: Extracted frame paths.
        """
        from ....core.exts import Exts
        from ....core.utils import get_temp_path
        from ...util import MediaConverter

        dst = get_temp_path(seed=str(src), suffix=Exts.PNG)
        base_dir = MediaConverter().extract_png_frames_from_video(
            src=src, dst=dst, frame_rate=self._fps
        )
        if base_dir is None:
            return []

        return sorted(base_dir.glob(f"{base_dir.stem}_*{Exts.PNG}"))

    def _extract_audio(self, src: Path) -> Optional[Path]:
        """Extract an audio track from a video.

        Args:
            src (Path): Video file path.

        Raises:
            ImportError: If ffmpeg is not installed.

        Returns:
            Optional[Path]: Extracted audio file path.
        """
        from ....core.exts import Exts
        from ....core.utils import get_temp_path
        from ...util import MediaConverter

        dst = Path(get_temp_path(seed=str(src), suffix=Exts.MP3))

        return MediaConverter().extract_mp3_audio_from_video(
            src=src, dst=dst, sample_rate=self._audio_sample_rate
        )

    def _image_docs(self, frames: Sequence[Path], source: str) -> list[Document]:
        """Convert frame images to Document objects.

        Args:
            frames (Sequence[Path]): Frame image paths.
            source (str): Source video path.

        Returns:
            list[Document]: Generated documents.
        """
        docs: list[Document] = []
        for i, frame_path in enumerate(frames):
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

    def lazy_load_data(self, path: Any, extra_info: Any = None) -> Iterable[Document]:
        """Split a video file into image and audio documents.

        Args:
            path (Any): Video file path-like object.
            extra_info (Any, optional): Unused extra info. Defaults to None.

        Returns:
            Iterable[Document]: Extracted documents.
        """
        from ....core.exts import Exts

        path = os.fspath(path)
        if not Exts.endswith_exts(path, set(Exts.VIDEO)):
            logger.error(
                f"unsupported video ext: {path}. supported: {' '.join(Exts.VIDEO)}"
            )
            return []

        try:
            frames = self._extract_frames(Path(path))
            audio = self._extract_audio(Path(path))
        except ImportError as e:
            logger.error(f"ffmpeg not installed, cannot read video files: {e}")
            return []

        docs = self._image_docs(frames=frames, source=path)
        if audio is not None:
            docs.append(self._audio_doc(audio, path))
            logger.debug(f"loaded {len(frames)} image docs + 1 audio doc from {path}")
        else:
            logger.debug(f"loaded {len(frames)} image docs from {path} (audio missing)")

        return docs
