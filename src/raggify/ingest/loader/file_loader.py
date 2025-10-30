from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ...config.default_settings import DefaultSettings as DS
from ...core.exts import Exts
from ...logger import logger
from .loader import Loader
from .reader.dummy_media_reader import DummyMediaReader
from .reader.pdf_reader import MultiPDFReader

if TYPE_CHECKING:
    from llama_index.core.readers.base import BaseReader
    from llama_index.core.schema import Document

    from ...vector_store.vector_store_manager import VectorStoreManager


class FileLoader(Loader):
    """ローカルファイルを読み込み、ドキュメントを生成するためのクラス。"""

    def __init__(
        self,
        store: VectorStoreManager,
        chunk_size: int = DS.CHUNK_SIZE,
        chunk_overlap: int = DS.CHUNK_OVERLAP,
    ) -> None:
        """コンストラクタ

        Args:
            store (VectorStoreManager): 登録済みソースの判定に使用
            chunk_size (int, optional): チャンクサイズ。Defaults to DS.CHUNK_SIZE.
            chunk_overlap (int, optional): チャンク重複語数。Defaults to DS.CHUNK_OVERLAP.
        """
        Loader.__init__(self, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self._store = store

        # 独自 reader の辞書。後段で SimpleDirectoryReader に渡す
        self._readers: dict[str, BaseReader] = {Exts.PDF: MultiPDFReader()}

        dummy_reader = DummyMediaReader()
        for ext in Exts.PASS_THROUGH_MEDIA:
            self._readers[ext] = dummy_reader

    async def aload_from_path(
        self,
        root: str,
    ) -> tuple[list[Document], list[Document], list[Document]]:
        """ローカルパス（ディレクトリ、ファイル）からコンテンツを取り込み、ドキュメントを生成する。
        ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

        Args:
            root (str): 対象パス

        Raises:
            ValueError: パスの指定誤り等

        Returns:
            tuple[list[Document], list[Document], list[Document]]:
                テキストドキュメント、画像ドキュメント、音声ドキュメント
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

        return self._split_docs_modality(docs)

    async def aload_from_paths(
        self,
        paths: list[str],
    ) -> tuple[list[Document], list[Document], list[Document]]:
        """パスリスト内の複数パスからコンテンツを取得し、ドキュメントを生成する。

        Args:
            paths (list[str]): パスリスト

        Returns:
            tuple[list[Document], list[Document], list[Document]]:
                テキストドキュメント、画像ドキュメント、音声ドキュメント
        """
        docs = []
        for path in paths:
            try:
                temp = await self.aload_from_path(path)
                docs.extend(temp)
            except Exception as e:
                logger.exception(e)
                continue

        return self._split_docs_modality(docs)
