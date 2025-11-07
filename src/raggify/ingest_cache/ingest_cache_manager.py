from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..embed.embed_manager import Modality

if TYPE_CHECKING:
    from llama_index.core.ingestion import IngestionCache


@dataclass
class IngestCacheStoreContainer:
    """モダリティ毎のインジェストキャッシュ関連パラメータを集約"""

    provider_name: str
    cache: Optional[IngestionCache]
    table_name: str


class IngestCacheStoreManager:
    """インジェストキャッシュの管理クラス。"""

    def __init__(self, conts: dict[Modality, IngestCacheStoreContainer]) -> None:
        """コンストラクタ

        Args:
            conts (dict[Modality, IngestCacheStoreContainer]): インジェストキャッシュコンテナの辞書
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
        """このインジェストキャッシュがサポートするモダリティ一覧。

        Returns:
            set[Modality]: モダリティ一覧
        """
        return set(self._conts.keys())

    def get_container(self, modality: Modality) -> IngestCacheStoreContainer:
        """モダリティ別のインジェストキャッシュコンテナを取得する。

        Args:
            modality (Modality): モダリティ

        Raises:
            RuntimeError: 未初期化

        Returns:
            IngestCacheStoreContainer: インジェストキャッシュコンテナ
        """
        cont = self._conts.get(modality)
        if cont is None:
            raise RuntimeError(f"store {modality} is not initialized")

        return cont
