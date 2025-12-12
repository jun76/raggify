from __future__ import annotations

import os
from typing import Any, Iterable

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ....logger import logger

__all__ = ["HTMLReader"]


class HTMLReader(BaseReader):
    """HTML file reader that extracts text content from HTML files.

    Note:
        Using SimpleWebPageReader causes two fetches to run during asset collection,
        so we reuse the HTML files saved to temporary files by this custom Reader.
    """

    def lazy_load_data(self, path: Any, extra_info: Any = None) -> Iterable[Document]:
        """Load an HTML file and generate text documents.

        Args:
            path (Any): File path-like object.

        Returns:
            Iterable[Document]: List of documents read from the HTML file.
        """
        from ....core.metadata import BasicMetaData

        try:
            path = os.fspath(path)
            with open(path, "r", encoding="utf-8") as f:
                import html2text

                html = f.read()

                # Convert to markdown-like text
                text = html2text.html2text(html)
        except OSError as e:
            logger.warning(f"failed to read HTML file {path}: {e}")
            return []

        metadata = BasicMetaData()
        doc = Document(text=text, metadata=metadata.to_dict())

        return [doc]
