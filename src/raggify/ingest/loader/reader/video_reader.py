from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ....core.exts import Exts
from ....core.metadata import BasicMetaData, get_temp_file_path_from
from ....logger import logger

__all__ = ["VideoReader"]


class VideoReader(BaseReader):
    """動画ファイルをフレーム画像と音声トラックに分解するためのリーダー。"""

    def __init__(
        self,
        *,
        fps: int = 1,
        audio_sample_rate: int = 16000,
        image_suffix: str = Exts.PNG,
        audio_suffix: str = Exts.WAV,
    ) -> None:
        """コンストラクタ。

        Args:
            fps (int, optional): 1 秒あたりの抽出フレーム数。Defaults to 1.
            audio_sample_rate (int, optional): 音声抽出時のサンプルレート。Defaults to 16000.
            image_suffix (str, optional): フレーム画像の拡張子。Defaults to Exts.PNG.
            audio_suffix (str, optional): 音声ファイルの拡張子。Defaults to Exts.WAV.

        Raises:
            ValueError: fps が 0 以下の場合
        """
        super().__init__()

        self._fps = fps
        self._audio_sample_rate = audio_sample_rate
        self._image_suffix = image_suffix
        self._audio_suffix = audio_suffix

    def _ffmpeg(self) -> Any:
        import ffmpeg  # type: ignore

        return ffmpeg

    def _extract_frames(self, src: str) -> list[Path]:
        """動画からフレーム画像を抽出する。

        Args:
            src (str): 動画ファイルのパス

        Returns:
            list[Path]: 抽出したフレームのパス
        """
        ffmpeg = self._ffmpeg()
        base_path = Path(get_temp_file_path_from(source=src, suffix=self._image_suffix))
        frames_dir = base_path.parent / f"{base_path.stem}_frames"

        if frames_dir.exists():
            shutil.rmtree(frames_dir)

        frames_dir.mkdir(parents=True, exist_ok=True)
        pattern = str(frames_dir / f"{base_path.stem}_%05d{self._image_suffix}")
        (
            ffmpeg.input(src)
            .filter("fps", self._fps)
            .output(pattern, format="image2", vcodec="png")
            .overwrite_output()
            .run(quiet=True)
        )
        frames = sorted(frames_dir.glob(f"{base_path.stem}_*{self._image_suffix}"))

        logger.debug(f"extracted {len(frames)} frame(s) from {src}")

        return frames

    def _extract_audio(self, src: str) -> Path | None:
        """動画から音声トラックを抽出する。

        Args:
            src (str): 動画ファイルのパス

        Returns:
            Path: 抽出した音声ファイルのパス
        """
        ffmpeg = self._ffmpeg()
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
        except ffmpeg.Error as err:  # type: ignore[attr-defined]
            logger.warning(f"ffmpeg audio extraction failure: {err}")
            try:
                temp_path.unlink()
            except OSError:
                pass
            return None

        return temp_path

    def _image_docs(self, frames: Sequence[Path], source: str) -> list[Document]:
        """フレーム画像を Document に変換する。

        Args:
            frames (Sequence[Path]): フレーム画像のパス一覧
            source (str): 元動画のパス

        Returns:
            list[Document]: 生成したドキュメント
        """
        docs: list[Document] = []
        for i, frame in enumerate(frames):
            meta = BasicMetaData()
            meta.file_path = str(frame)
            meta.temp_file_path = str(frame)
            meta.base_source = source
            meta.page_no = i
            docs.append(Document(text=source, metadata=meta.to_dict()))

        return docs

    def _audio_doc(self, audio_path: Path, source: str) -> Document:
        """音声ファイルを Document に変換する。

        Args:
            audio_path (Path): 音声ファイルのパス
            source (str): 元動画のパス

        Returns:
            Document: 生成した音声ドキュメント
        """
        meta = BasicMetaData()
        meta.file_path = str(audio_path)
        meta.temp_file_path = str(audio_path)
        meta.base_source = source

        return Document(text=source, metadata=meta.to_dict())

    def _load_video(self, path: str, allowed_exts: Iterable[str]) -> list[Document]:
        """動画を読み込み、フレーム＋音声のドキュメントを生成する。

        Args:
            path (str): 動画ファイルパス
            allowed_exts (Iterable[str]): 許可する拡張子

        Raises:
            ValueError: 許可されていない拡張子が指定された場合

        Returns:
            list[Document]: 生成したドキュメント
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
        """動画ファイルを画像＋音声ドキュメントに分解する。

        Args:
            path (str): 動画ファイルパス
            extra_info (Any, optional): 追加情報（未使用）。Defaults to None.

        Returns:
            Iterable[Document]: 抽出したドキュメント
        """
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            logger.warning(f"file not found: {abs_path}")
            return []

        return self._load_video(abs_path, allowed_exts=Exts.VIDEO)
