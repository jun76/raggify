from __future__ import annotations

import os
from urllib.parse import urlparse


class Exts:
    # 個別参照用
    PNG: str = ".png"
    WAV: str = ".wav"
    PDF: str = ".pdf"
    MP4: str = ".mp4"

    # 基本的に reader(llama_index.core.readers.file.base._try_loading_included_file_formats)
    # のサポートする拡張子に追従する。
    # ただし、reader はその他の拡張子もフォールバックとしてテキストファイル扱いで
    # 読み込もうとするため、逆に .txt 等のテキストファイルの拡張子は明記されていない点に注意。

    # base64 エンコーディングしてマルチモーダル（画像）の埋め込みモデルに渡せる拡張子
    IMAGE: set[str] = {".gif", ".jpg", PNG, ".jpeg", ".webp"}

    # マルチモーダル（音声）の埋め込みモデルに渡せる拡張子
    AUDIO: set[str] = {WAV, ".flac", ".ogg", ".mp3"}

    # マルチモーダル（動画）の埋め込みモデルに渡せる拡張子
    VIDEO: set[str] = {".wmv", MP4, ".avi"}

    # サイトマップの抽出判定に使用する拡張子
    SITEMAP: set[str] = {".xml"}

    # Web ページから予想外のファイルや巨大な動画ファイルをフェッチしてこないように絞る
    _DEFAULT_FETCH_TARGET: set[str] = {  # うち、専用の reader が存在するもの
        ".hwp",
        PDF,
        ".docx",
        ".pptx",
        ".ppt",
        ".pptm",
        ".csv",
        ".epub",
        ".mbox",
        ".ipynb",
        ".xls",
        ".xlsx",
    }
    _ADDITIONAL_FETCH_TARGET: set[str] = {  # その他にフェッチしたいもの
        ".txt",
        ".text",
        ".md",
        ".json",
    }
    FETCH_TARGET: set[str] = (
        IMAGE
        | AUDIO
        | VIDEO
        | SITEMAP
        | _DEFAULT_FETCH_TARGET
        | _ADDITIONAL_FETCH_TARGET
    )

    # 専用 reader に処理させずにファイルパスのみを素通しさせておき、
    # 後段の upsert 時に埋め込みモデルに直接処理させたい拡張子セット
    PASS_THROUGH_MEDIA = AUDIO | VIDEO

    @classmethod
    def endswith_exts(cls, s: str, exts: set[str]) -> bool:
        """文字列の末尾に指定の拡張子が含まれるか。

        Args:
            s (str): 文字列
            exts (set[str]): チェック対象の拡張子セット

        Returns:
            bool: 含まれる場合 True
        """
        return any(s.lower().endswith(ext) for ext in exts)

    @classmethod
    def endswith_ext(cls, s: str, ext: str) -> bool:
        """文字列の末尾に指定の拡張子が含まれるか。

        Args:
            s (str): 文字列
            ext (str): チェック対象の拡張子

        Returns:
            bool: 含まれる場合 True
        """
        return s.lower().endswith(ext.lower())

    @classmethod
    def get_ext(cls, uri: str, dot: bool = True) -> str:
        """ファイルパスまたは URL 文字列から拡張子を取得する。

        Args:
            uri (str): ファイルパスまたは URL
            dot (bool, optional): ドットをつけるか。Defaults to True.

        Returns:
            str: 拡張子
        """
        parsed = urlparse(uri)
        ext = os.path.splitext(parsed.path)[1].lower()

        if not dot:
            return ext.replace(".", "")

        return ext
