from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..embed.embed_manager import Modality

if TYPE_CHECKING:
    from llama_index.core.ingestion import IngestionCache


@dataclass
class IngestCacheStoreContainer:
    """モダリティ毎のキャッシュストア関連パラメータを集約"""

    provider_name: str
    store: Optional[IngestionCache]
    table_name: str


class IngestCacheStoreManager:
    """キャッシュストアの管理クラス。"""

    def __init__(self, conts: dict[Modality, IngestCacheStoreContainer]) -> None:
        """コンストラクタ

        Args:
            conts (dict[Modality, IngestCacheStoreContainer]): キャッシュストアコンテナの辞書
        """
        self._conts = conts

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        return ", ".join([cont.provider_name for cont in self._conts.values()])

    @property
    def modality(self) -> set[Modality]:
        """このキャッシュストアがサポートするモダリティ一覧。

        Returns:
            set[Modality]: モダリティ一覧
        """
        return set(self._conts.keys())

    def get_container(self, modality: Modality) -> IngestCacheStoreContainer:
        """モダリティ別のキャッシュストアコンテナを取得する。

        Args:
            modality (Modality): モダリティ

        Raises:
            RuntimeError: 未初期化

        Returns:
            IngestCacheStoreContainer: キャッシュストアコンテナ
        """
        cont = self._conts.get(modality)
        if cont is None:
            raise RuntimeError(f"store {modality} is not initialized")

        return cont
