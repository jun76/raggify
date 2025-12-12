from __future__ import annotations

import os
from typing import Any, Iterable

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ....logger import logger

__all__ = ["DummyMediaReader"]


class DummyMediaReader(BaseReader):
    """Dummy reader to prevent downstream default readers from splitting media as text."""

    def lazy_load_data(self, path: Any, extra_info: Any = None) -> Iterable[Document]:
        """Load media files as dummy documents containing file paths.

        Args:
            path (Any): File path-like object.

        Returns:
            Iterable[Document]: Documents containing the file path.
        """
        from ....core.exts import Exts
        from ....core.metadata import MetaKeys as MK

        path = os.fspath(path)
        ext = Exts.get_ext(path)
        if ext not in Exts.PASS_THROUGH_MEDIA:
            logger.warning(
                f"unsupported ext {ext}: {' '.join(Exts.PASS_THROUGH_MEDIA)} are allowed."
            )
            return []

        # For MultiModalVectorStoreIndex
        doc = Document(text=path, metadata={MK.FILE_PATH: path})

        logger.debug(f"loaded 1 doc from {path}")

        return [doc]
