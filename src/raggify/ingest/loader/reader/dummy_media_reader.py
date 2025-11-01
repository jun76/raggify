from __future__ import annotations

import os
from typing import Any, Iterable

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ....core.exts import Exts
from ....logger import logger


class DummyMediaReader(BaseReader):
    """後段のデフォルトリーダーにテキストとして解釈・スプリットさせないためのダミーリーダー"""

    def lazy_load_data(self, path: str, extra_info: Any = None) -> Iterable[Document]:
        """メディアファイルをダミーとして読み込み、ファイルパスを含むドキュメントを生成する。

        Args:
            path (str): ファイルパス

        Returns:
            Iterable[Document]: テキストドキュメントと画像ドキュメント
        """
        from ....core.metadata import MetaKeys as MK

        path = os.path.abspath(path)
        if not os.path.exists(path):
            logger.warning(f"file not found: {path}")
            return []

        if not Exts.endswith_exts(path, Exts.PASS_THROUGH_MEDIA):
            logger.warning(
                f"unsupported ext for {path}. {' '.join(Exts.PASS_THROUGH_MEDIA)} is allowed."
            )
            return []

        # MultiModalVectorStoreIndex 参照用
        doc = Document(text=path, metadata={MK.FILE_PATH: path})

        logger.debug(f"loaded 1 text doc from {path}")

        return [doc]
