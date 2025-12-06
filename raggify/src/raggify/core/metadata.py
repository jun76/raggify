from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

__all__ = ["MetaKeysFrom", "MetaKeys", "BasicMetaData"]


class MetaKeysFrom:
    # Labels defined by the library (do not change strings)
    ## SimpleDirectoryReader
    FILE_PATH = "file_path"
    FILE_TYPE = "file_type"
    FILE_SIZE = "file_size"
    FILE_CREATED_AT = "creation_date"
    FILE_LASTMOD_AT = "last_modified_date"


class MetaKeys(MetaKeysFrom):
    # Normalized labels added by the app
    CHUNK_NO = "chunk_no"
    URL = "url"
    BASE_SOURCE = "base_source"
    TEMP_FILE_PATH = "temp_file_path"
    PAGE_NO = "page_no"
    ASSET_NO = "asset_no"


@dataclass(kw_only=True)
class BasicMetaData:
    """Metadata container for document/node `metadata` fields.

    Uses fields auto-attached by readers and defines additional fields explicitly
    inserted and consumed by the app.

    Reference
        SimpleDirectoryReader:
            file_path
            file_name
            file_type
            file_size
            creation_date
            last_modified_date
            last_accessed_date

    Downstream readers (implementation-specific)
        PDFReader:
            page_label

        PptxReader:
            file_path
            page_label
            title
            extraction_errors
            extraction_warnings
            tables
            charts
            notes
            images
            text_sections

        ImageReader:
            Various merged metadata from lower readers
    """

    # Metadata payload.
    # When adding/removing fields, keep consistency with loaders creating node instances
    # and metadata-store implementations managing metadata.
    #
    file_path: str = ""  # Source file path
    file_type: str = ""  # File type (mimetype)
    file_size: int = 0  # File size
    file_created_at: str = ""  # File creation timestamp
    file_lastmod_at: str = ""  # Last modified timestamp
    chunk_no: int = 0  # Chunk number of text
    url: str = ""  # Source URL
    base_source: str = ""  # Origin info (e.g., parent page of direct-linked image)
    temp_file_path: str = ""  # Temporary file path for downloaded assets
    page_no: int = 0  # Page number
    asset_no: int = 0  # Asset number (e.g., image within the same page)

    @classmethod
    def from_dict(cls, meta: Optional[dict[str, Any]] = None) -> "BasicMetaData":
        """Create a metadata instance from a dict.

        Args:
            meta (Optional[dict[str, Any]], optional): Metadata dict. Defaults to None.

        Returns:
            BasicMetaData: Generated metadata instance.
        """
        data = meta or {}

        return cls(
            file_path=data.get(MetaKeys.FILE_PATH, ""),
            file_type=data.get(MetaKeys.FILE_TYPE, ""),
            file_size=data.get(MetaKeys.FILE_SIZE, 0),
            file_created_at=data.get(MetaKeys.FILE_CREATED_AT, ""),
            file_lastmod_at=data.get(MetaKeys.FILE_LASTMOD_AT, ""),
            chunk_no=data.get(MetaKeys.CHUNK_NO, 0),
            url=data.get(MetaKeys.URL, ""),
            base_source=data.get(MetaKeys.BASE_SOURCE, ""),
            temp_file_path=data.get(MetaKeys.TEMP_FILE_PATH, ""),
            page_no=data.get(MetaKeys.PAGE_NO, 0),
            asset_no=data.get(MetaKeys.ASSET_NO, 0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return metadata as a dict.

        Returns:
            dict[str, Any]: Metadata dict.
        """
        return asdict(self)
