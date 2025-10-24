from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from ...config.default_settings import DefaultSettings as DS
from ...core.exts import Exts
from ...logger import logger
from .loader import Loader
from .reader.dummy_media_reader import DummyMediaReader
from .reader.pdf_reader import MultiPDFReader

if TYPE_CHECKING:
    from llama_index.core.node_parser.interface import MetadataAwareTextSplitter
    from llama_index.core.readers.base import BaseReader
    from llama_index.core.readers.file.base import SimpleDirectoryReader
    from llama_index.core.schema import BaseNode

    from ...vector_store.vector_store_manager import VectorStoreManager


class FileLoader(Loader):
    def __init__(
        self,
        store: VectorStoreManager,
        chunk_size: int = DS.CHUNK_SIZE,
        chunk_overlap: int = DS.CHUNK_OVERLAP,
    ) -> None:
        """ローカルファイルを読み込み、ノードを生成するためのクラス。

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

    async def _aload_from_file(
        self,
        path: str,
        reader: SimpleDirectoryReader,
        splitter: MetadataAwareTextSplitter,
    ) -> list[BaseNode]:
        """ファイルからノードを生成する。

        Args:
            path (str): ファイルパス
            reader (SimpleDirectoryReader): ファイルリーダ
            splitter (MetadataAwareTextSplitter): テキストスプリッタ

        Raises:
            RuntimeError: ノードの生成に失敗

        Returns:
            list[BaseNode]: 生成したノード
        """
        from ...core.metadata import BasicMetaData

        try:
            docs = await reader.aload_file(
                input_file=Path(path),
                file_metadata=reader.file_metadata,
                file_extractor=reader.file_extractor,
            )

            all_nodes = []
            for doc in docs:
                nodes = await splitter.aget_nodes_from_documents([doc])
                for i, node in enumerate(nodes):
                    meta = BasicMetaData().from_dict(node.metadata)
                    meta.chunk_no = i
                    meta.node_lastmod_at = time.time()
                    node.metadata = meta.to_dict()

                all_nodes.extend(nodes)
        except Exception as e:
            raise RuntimeError(f"failed to generate nodes from file: {path}") from e

        logger.debug(f"loaded {len(all_nodes)} nodes from {path}")

        return all_nodes

    async def aload_from_path(
        self,
        root: str,
    ) -> list[BaseNode]:
        """ローカルパス（ディレクトリ、ファイル）からコンテンツを取り込み、ノードを生成する。
        ディレクトリの場合はツリーを下りながら複数ファイルを取り込む。

        Args:
            root (str): 対象パス

        Raises:
            ValueError: パスの指定誤り等

        Returns:
            list[BaseNode]: 生成したノード
        """
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.readers.file.base import SimpleDirectoryReader

        try:
            path = Path(root).absolute()
            reader = SimpleDirectoryReader(
                input_dir=root if path.is_dir() else None,
                input_files=[root] if path.is_file() else None,
                recursive=True,
                file_extractor=self._readers,
            )

            splitter = SentenceSplitter(
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                include_metadata=True,
            )

            paths = reader.list_resources()
        except Exception as e:
            logger.exception(e)
            raise ValueError("failed to load from path") from e

        # 最上位ループ内で複数ソースをまたいで _source_cache を共有したいため
        # ここでは _source_cache.clear() しないこと。
        nodes = []
        for path in paths:
            try:
                if path in self._source_cache:
                    continue

                if self._store.skip_update(path):
                    logger.debug(f"skip loading: source exists ({path})")
                    continue

                nodes.extend(
                    await self._aload_from_file(
                        path=path, reader=reader, splitter=splitter
                    )
                )

                # 取得済みキャッシュに追加
                self._source_cache.add(path)
            except Exception as e:
                logger.exception(e)
                continue

        logger.debug(f"loaded {len(nodes)} nodes from {root}")

        return nodes

    async def aload_from_paths(
        self,
        paths: list[str],
    ) -> list[BaseNode]:
        """パスリスト内の複数パスからコンテンツを取得し、ノードを生成する。

        Args:
            paths (list[str]): パスリスト

        Returns:
            list[BaseNode]: 生成したノード
        """
        # 最上位ループ。キャッシュを空にしてから使う。
        self._source_cache.clear()
        nodes = []
        for path in paths:
            try:
                temp = await self.aload_from_path(path)
                nodes.extend(temp)
            except Exception as e:
                logger.exception(e)
                continue

        return nodes
