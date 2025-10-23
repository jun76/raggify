from __future__ import annotations

import os
import time
from typing import Any, Iterable

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ....core.exts import Exts
from ....logger import logger


class DummyMediaReader(BaseReader):
    """後段のデフォルトリーダーにテキストとして解釈・スプリットさせないためのダミーリーダー"""

    def lazy_load_data(self, path: str, extra_info: Any = None) -> Iterable[Document]:
        """PDF ファイルを読み込み、テキストと画像のドキュメントをそれぞれ生成する。

        Args:
            path (str): ファイルパス

        Returns:
            Iterable[Document]: テキストドキュメントと画像ドキュメント
        """
        from ....core.metadata import BasicMetaData

        path = os.path.abspath(path)
        if not os.path.exists(path):
            logger.warning(f"file not found: {path}")
            return []

        if not Exts.endswith_exts(path, Exts.DUMMY_MEDIA):
            logger.warning(f"unsupported ext. {' '.join(Exts.DUMMY_MEDIA)} is allowed.")
            return []

        meta = BasicMetaData()
        meta.file_path = path  # MultiModalVectorStoreIndex 参照用
        meta.node_lastmod_at = time.time()

        doc = Document(text=path, metadata=meta.to_dict())

        logger.info(f"loaded 1 text doc from {path}")

        return [doc]
