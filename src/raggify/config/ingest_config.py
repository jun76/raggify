from __future__ import annotations

from dataclasses import dataclass

from .default_settings import DefaultSettings


@dataclass(kw_only=True)
class IngestConfig:
    """ドキュメント取り込み処理関連の設定用データクラス"""

    # General
    chunk_size: int = DefaultSettings.CHUNK_SIZE
    chunk_overlap: int = DefaultSettings.CHUNK_OVERLAP
    upload_dir: str = DefaultSettings.UPLOAD_DIR
    pipe_persist_dir: str = DefaultSettings.PIPE_PERSIST_DIR

    # Web
    user_agent: str = DefaultSettings.USER_AGENT
    load_asset: bool = DefaultSettings.LOAD_ASSET
    req_per_sec: int = DefaultSettings.REQ_PER_SEC
    timeout_sec: int = DefaultSettings.TIMEOUT_SEC
    same_origin: bool = DefaultSettings.SAME_ORIGIN
