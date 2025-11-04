from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


class MetaKeysFrom:
    # ライブラリ側定義ラベル（字列変更不可）
    ## SimpleDirectoryReader
    FILE_PATH = "file_path"
    FILE_TYPE = "file_type"
    FILE_SIZE = "file_size"
    FILE_CREATED_AT = "creation_date"
    FILE_LASTMOD_AT = "last_modified_date"


class MetaKeys(MetaKeysFrom):
    # 正規化し、アプリ側で付与するラベル
    CHUNK_NO = "chunk_no"
    URL = "url"
    BASE_SOURCE = "base_source"
    TEMP_FILE_PATH = "temp_file_path"
    PAGE_NO = "page_no"
    ASSET_NO = "asset_no"


@dataclass
class BasicMetaData:
    """ドキュメント、ノードの metadata フィールド用。
    Reader が自動付与するものを利用しつつ、アプリ側で明示的に挿入・利用するものはここで定義。

    参考
        SimpleDirectoryReader:
            file_path
            file_name
            file_type
            file_size
            creation_date
            last_modified_date
            last_accessed_date

    後段の各 Reader（基本、実装依存）
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
            下位 Reader 独自メタを合流する形のため色々

        等
    """

    # メタデータの中身
    # 追加・削除する場合、ノードのインスタンスを生成する loader 系の実装と
    # メタ情報を管理する meta_store 系の実装とも整合させること
    #
    file_path: str = ""  # 取得元ファイルパス
    file_type: str = ""  # ファイル種別（mimetype）
    file_size: int = 0  # ファイルサイズ
    file_created_at: str = ""  # ファイル作成日時
    file_lastmod_at: str = ""  # 最終更新日時
    chunk_no: int = 0  # テキストのチャンク番号
    url: str = ""  # 取得元 URL
    base_source: str = ""  # 出典情報（直リンク画像の親ページ等）
    temp_file_path: str = ""  # ダウンロード画像等の一時ファイルパス
    page_no: int = 0  # ページ番号
    asset_no: int = 0  # アセット番号（同一ページ内の画像等）

    @classmethod
    def from_dict(cls, meta: Optional[dict[str, Any]] = None) -> "BasicMetaData":
        """dict からメタデータインスタンスを生成する。

        Args:
            meta (Optional[dict[str, Any]], optional): メタデータの dict。 Defaults to None.
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
        """メタデータの dict を返す。

        Returns:
            dict[str, Any]: メタデータの dict
        """
        return asdict(self)


def get_temp_file_path_from(source: str, suffix: str) -> str:
    """ソース情報に一意に対応付く一時ファイルパスを取得する。

    PDF ファイルから抽出した画像ファイル等の管理用途を想定。
    ランダムな字列だと metadata に含まれた時に hash を揺らす原因となるため。

    Args:
        source (str): パスまたは URL。ページ番号等がなければ区別できない場合はそれも付加
        suffix (str): 拡張子等

    Returns:
        str: 一時ファイルパス
    """
    import hashlib
    import tempfile
    from pathlib import Path

    temp_dir = Path(tempfile.gettempdir())
    filename = hashlib.md5(source.encode()).hexdigest() + suffix

    return str(temp_dir / filename)
