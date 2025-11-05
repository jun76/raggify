from __future__ import annotations

__all__ = ["sanitize_str"]


def sanitize_str(s: str, hash: bool = False) -> str:
    """様々な制約を考慮した、無難な文字列を生成する。

    原則として、何か新たな制約が加わるたびにこのインターフェースに盛り込んで
    caller 全員に従わせる。従えない場合はこのインターフェースの使用を止めて
    個別にサニタイズを行う。

    Args:
        s (str): 入力文字列
        hash (bool, optional): 長すぎた時にハッシュ文字列化するか。Defaults to False.

    Raises:
        ValueError: 長すぎる文字列

    Returns:
        str: サニタイズ後の文字列

    Note:
        既知の制約(AND):
            Chroma
                containing 3-512 characters from [a-zA-Z0-9._-],
                starting and ending with a character in [a-zA-Z0-9]
            PGVector
                maximum length of 63 characters
    """
    import re

    MIN_LEN = 3
    MAX_LEN = 63

    # 記号は全てアンダースコアに置換
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", s)

    l = len(sanitized)
    if l < MIN_LEN:
        # 不足時はアンダースコアで埋める
        return f"{sanitized:_>{MIN_LEN}}"

    if l > MAX_LEN:
        # 超過時は
        if hash:
            # hash 文字列化するか
            import hashlib

            return hashlib.md5(sanitized.encode()).hexdigest()
        else:
            # raise するか
            raise ValueError(f"too long string: {sanitized} > {MAX_LEN}")

    return sanitized
