from __future__ import annotations

__all__ = ["sanitize_str"]


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
