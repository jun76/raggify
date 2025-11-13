from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from ...core.exts import Exts
from ...logger import logger
from .loader import Loader
from .reader.dummy_media_reader import DummyMediaReader
from .reader.pdf_reader import MultiPDFReader

if TYPE_CHECKING:
    from llama_index.core.readers.base import BaseReader
    from llama_index.core.schema import ImageNode, TextNode

    from ...llama.core.schema import AudioNode, VideoNode


class FileLoader(Loader):
    """ローカルファイルを読み込み、ノードを生成するためのクラス。"""

    def __init__(self, persist_dir: Optional[Path]) -> None:
        """コンストラクタ

        Args:
            persist_dir (Optional[Path]): 永続化ディレクトリ
        """
        super().__init__(persist_dir)

        # 独自 reader の辞書。後段で SimpleDirectoryReader に渡す
        self._readers: dict[str, BaseReader] = {Exts.PDF: MultiPDFReader()}

        dummy_reader = DummyMediaReader()
        for ext in Exts.PASS_THROUGH_MEDIA:
            self._readers[ext] = dummy_reader

    async def aload_from_path(
        self,
        root: str,
        is_canceled: Callable[[], bool],
    ) -> tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
        """ローカルパス（ディレクトリ、ファイル）からコンテンツを取り込み、ノードを生成する。
        ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

        Args:
            root (str): 対象パス
            is_canceled (Callable[[], bool]): このジョブがキャンセルされたか。

        Raises:
            ValueError: パスの指定誤り等

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
                テキストノード、画像ノード、音声ノード、動画ノード
        """
        from llama_index.core.readers.file.base import SimpleDirectoryReader

        try:
            path = Path(root).absolute()
            reader = SimpleDirectoryReader(
                input_dir=root if path.is_dir() else None,
                input_files=[root] if path.is_file() else None,
                recursive=True,
                file_extractor=self._readers,
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
        """パスリスト内の複数パスからコンテンツを取得し、ノードを生成する。

        Args:
            paths (list[str]): パスリスト
            is_canceled (Callable[[], bool]): このジョブがキャンセルされたか。

        Returns:
            tuple[list[TextNode], list[ImageNode], list[AudioNode], list[VideoNode]]:
                テキストノード、画像ノード、音声ノード、動画ノード
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
