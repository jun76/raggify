from __future__ import annotations

from dataclasses import dataclass

from .default_settings import DefaultSettings


@dataclass(kw_only=True)
class MetaStoreConfig:
    """メタ情報ストア関連の設定用データクラス"""

    meta_store_path: str = DefaultSettings.META_STORE_PATH
