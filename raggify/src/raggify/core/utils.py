from __future__ import annotations

from llama_index.core.schema import BaseNode

from .exts import Exts
from .metadata import MetaKeys as MK

__all__ = ["sanitize_str", "get_temp_file_path_from", "has_media"]


def sanitize_str(s: str, hash: bool = False) -> str:
    """Generate a safe string considering various constraints.

    In principle, add new constraints here so all callers comply.
    If a caller cannot, stop using this helper and sanitize separately.

    Args:
        s (str): Input string.
        hash (bool, optional): Hash the string when it is too long. Defaults to False.

    Raises:
        ValueError: If the string is too long.

    Returns:
        str: Sanitized string.

    Note:
        Known constraints (AND):
            Chroma
                containing 3-512 characters from [a-zA-Z0-9._-],
                starting and ending with a character in [a-zA-Z0-9]
            PGVector
                maximum length of 63 characters
    """
    import re

    MIN_LEN = 3
    MAX_LEN = 63

    # Replace all symbols with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", s)

    l = len(sanitized)
    if l < MIN_LEN:
        # Pad with underscores if too short
        return f"{sanitized:_>{MIN_LEN}}"

    if l > MAX_LEN:
        # Too long
        if hash:
            # Hash the string
            import hashlib

            return hashlib.md5(sanitized.encode()).hexdigest()
        else:
            # Raise error
            raise ValueError(f"too long string: {sanitized} > {MAX_LEN}")

    return sanitized


def get_temp_file_path_from(source: str, suffix: str) -> str:
    """Get a temporary file path uniquely tied to the source.

    Intended for managing assets extracted from PDFs, etc. Avoid random strings
    so hashes stay stable when metadata contains the path.

    Args:
        source (str): Path or URL. Include page numbers, etc., if needed for uniqueness.
        suffix (str): Extension or suffix.

    Returns:
        str: Temporary file path.
    """
    import hashlib
    import tempfile
    from pathlib import Path

    from .const import TEMP_FILE_PREFIX

    temp_dir = Path(tempfile.gettempdir())
    filename = TEMP_FILE_PREFIX + hashlib.md5(source.encode()).hexdigest() + suffix

    return str(temp_dir / filename)


def has_media(node: BaseNode, exts: set[str]) -> bool:
    """Return True if the node has media extensions.

    Args:
        node (BaseNode): Target node.
        exts (set[str]): Extension set.

    Returns:
        bool: True if matched.
    """
    path = node.metadata.get(MK.FILE_PATH, "")
    url = node.metadata.get(MK.URL, "")

    # Include those whose temp_file_path
    # (via custom readers) contains relevant extensions
    temp_file_path = node.metadata.get(MK.TEMP_FILE_PATH, "")

    return (
        Exts.endswith_exts(path, exts)
        or Exts.endswith_exts(url, exts)
        or Exts.endswith_exts(temp_file_path, exts)
    )
