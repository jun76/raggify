from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Iterable

from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import Document

from ....core.const import EXTRA_PKG_NOT_FOUND_MSG
from ....logger import logger

_PYMUPDF_NOT_FOUND_MSG = EXTRA_PKG_NOT_FOUND_MSG.format(
    pkg="pymupdf",
    extra="image",
    feature="MultiPDFReader",
)

if TYPE_CHECKING:
    try:
        from fitz import Document as FDoc  # type: ignore
    except ImportError:
        raise ImportError(_PYMUPDF_NOT_FOUND_MSG)

__all__ = ["MultiPDFReader"]


class MultiPDFReader(BaseReader):
    """Custom PDF reader that also extracts images."""

    def lazy_load_data(self, path: Any, extra_info: Any = None) -> Iterable[Document]:
        """Load a PDF file and generate text and image documents.

        Args:
            path (Any): File path-like object.

        Raises:
            ImportError: If pymupdf is not installed.

        Returns:
            Iterable[Document]: Text and image documents.
        """
        from ....core.exts import Exts

        try:
            import pymupdf as fitz  # type: ignore
        except ImportError:
            raise ImportError(_PYMUPDF_NOT_FOUND_MSG)

        path = os.fspath(path)
        if not Exts.endswith_ext(path, Exts.PDF):
            logger.warning(f"unsupported ext. {Exts.PDF} is allowed.")
            return []

        try:
            pdf = fitz.open(path)
        except Exception as e:
            logger.exception(e)
            return []

        try:
            text_docs = self._load_pdf_texts(pdf, path)
            image_docs = self._load_pdf_images(pdf, path)
        finally:
            pdf.close()

        logger.debug(
            f"loaded {len(text_docs)} text docs, {len(image_docs)} image docs from {path}"
        )

        return text_docs + image_docs

    def _load_pdf_texts(
        self,
        pdf: FDoc,
        path: str,
    ) -> list[Document]:
        """Generate documents from texts of a PDF.

        Args:
            pdf (FDoc): PDF instance.
            path (str): File path.

        Returns:
            list[Document]: Generated documents.
        """
        from ....core.metadata import BasicMetaData

        docs = []
        for page_no in range(pdf.page_count):
            try:
                page = pdf.load_page(page_no)
                content = page.get_text("text")  # type: ignore
            except Exception as e:
                logger.exception(e)
                continue

            # Skip if empty
            if not content.strip():  # type: ignore
                continue

            meta = BasicMetaData()
            meta.file_path = path
            meta.page_no = page_no

            doc = Document(text=content, metadata=meta.to_dict())
            docs.append(doc)

        return docs

    def _load_pdf_images(
        self,
        pdf: FDoc,
        path: str,
    ) -> list[Document]:
        """Generate documents from images of a PDF.

        Args:
            pdf (FDoc): PDF instance.
            path (str): File path.

        Raises:
            ImportError: If pymupdf is not installed.

        Returns:
            list[Document]: Generated documents.
        """
        try:
            import pymupdf as fitz  # type: ignore
        except ImportError:
            raise ImportError(_PYMUPDF_NOT_FOUND_MSG)

        from ....core.exts import Exts
        from ....core.metadata import BasicMetaData
        from ....core.utils import get_temp_path

        docs = []
        for page_no in range(pdf.page_count):
            try:
                page = pdf.load_page(page_no)
                contents = page.get_images(full=True)  # type: ignore
            except Exception as e:
                logger.exception(e)
                continue

            for image_no, image in enumerate(contents):
                xref = image[0]  # Image reference number
                pix = None
                try:
                    pix = fitz.Pixmap(pdf, xref)

                    if (
                        pix.n - (1 if pix.alpha else 0) == 4
                    ):  # CMYK (regardless of alpha)
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    temp = str(
                        get_temp_path(
                            seed=f"{path}:{page_no}:{image_no}", suffix=Exts.PNG
                        )
                    )
                    pix.save(temp)

                    meta = BasicMetaData()
                    meta.file_path = temp  # For MultiModalVectorStoreIndex
                    meta.temp_file_path = temp  # For cleanup
                    meta.base_source = path  # For restoring original path
                    meta.page_no = page_no
                    meta.asset_no = image_no

                    doc = Document(text="", metadata=meta.to_dict())
                    docs.append(doc)
                except Exception as e:
                    logger.exception(e)
                    continue
                finally:
                    if pix is not None:
                        del pix

        return docs
