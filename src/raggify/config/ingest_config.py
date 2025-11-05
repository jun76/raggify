from __future__ import annotations

from dataclasses import dataclass

from ..core.const import DEFAULT_KNOWLEDGEBASE_NAME, PROJECT_NAME


@dataclass(kw_only=True)
class IngestConfig:
    """ドキュメント取り込み処理関連の設定用データクラス"""

    # General
    chunk_size: int = 500
    chunk_overlap: int = 50
    upload_dir: str = f"/etc/{PROJECT_NAME}/upload"
    pipe_persist_dir: str = f"/etc/{PROJECT_NAME}/{DEFAULT_KNOWLEDGEBASE_NAME}"

    # Web
    user_agent: str = PROJECT_NAME
    load_asset: bool = True
    req_per_sec: int = 2
    timeout_sec: int = 30
    same_origin: bool = True
