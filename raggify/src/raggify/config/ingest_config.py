from __future__ import annotations

from dataclasses import dataclass

from .default_settings import DefaultSettings


@dataclass(kw_only=True)
class IngestConfig:
    """ドキュメント取り込み処理関連の設定用データクラス"""

    chunk_size: int = DefaultSettings.CHUNK_SIZE
    chunk_overlap: int = DefaultSettings.CHUNK_OVERLAP
    user_agent: str = DefaultSettings.USER_AGENT
    upload_dir: str = DefaultSettings.UPLOAD_DIR
