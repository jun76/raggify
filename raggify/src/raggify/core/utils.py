from __future__ import annotations

from pathlib import Path
from typing import Optional

from llama_index.core.schema import BaseNode

from .exts import Exts
from .metadata import MetaKeys as MK

__all__ = ["sanitize_str", "get_temp_path", "has_media"]


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

    sanitized_len = len(sanitized)
    if sanitized_len < MIN_LEN:
        # Pad with underscores if too short
        return f"{sanitized:_>{MIN_LEN}}"

    if sanitized_len > MAX_LEN:
        # Too long
        if hash:
            # Hash the string
            import hashlib

            return hashlib.md5(sanitized.encode()).hexdigest()
        else:
            # Raise error
            raise ValueError(f"too long string: {sanitized} > {MAX_LEN}")

    return sanitized


def get_temp_path(seed: str, suffix: Optional[str] = None) -> Path:
    """Get a temporary file or directory path uniquely tied to the source.

    Intended for managing assets extracted from PDFs, etc. Avoid random strings
    so hashes stay stable when metadata contains the path.

    Args:
        seed (str): Path or URL. Include page numbers, etc., if needed for uniqueness.
        suffix (Optional[str], optional): Extension or suffix. Defaults to None.

    Returns:
        Path: Temporary file or directory path.
    """
    import hashlib
    import tempfile
    from pathlib import Path

    from .const import TEMP_FILE_PREFIX

    temp_dir = Path(tempfile.gettempdir())
    base_name = TEMP_FILE_PREFIX + hashlib.md5(seed.encode()).hexdigest()
    if suffix is None:
        temp_path = temp_dir / base_name
    else:
        filename = base_name + suffix
        temp_path = temp_dir / filename

    return temp_path


def make_temp_dir(path: Path, remove: bool = True) -> None:
    """Create a temporary directory, optionally removing existing one.

    Args:
        path (Path): Directory or file path.
        remove (bool, optional): Remove existing directory. Defaults to True.
    """
    import shutil

    # path.suffix:
    # - Heuristic: treat a path with an extension as a file path.
    # - Works even when the path does not exist yet.
    # - Example: tmp_raggify_xxx.mp3 has a suffix, so it is treated as a file.
    # path.is_file():
    # - Checks whether the path currently exists as a file.
    # - Always False for non-existent paths.
    # - Example: tmp_raggify_xxx.mp3 returns False if it has not been created yet.
    #
    # Use suffix to detect file-intent, and is_file only for existing paths.
    if path.suffix:
        if remove and path.exists():
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        return

    if path.is_file():
        path = path.parent

    if remove and path.exists():
        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


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
