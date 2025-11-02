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

    def __init__(
        self,
        conts: dict[Modality, IngestCacheStoreContainer],
        persist_path: Optional[str] = None,
    ) -> None:
        """コンストラクタ

        Args:
            conts (dict[Modality, IngestCacheStoreContainer]): キャッシュストアコンテナの辞書
            persist_path (Optional[str], optional): 永続化パス。Defaults to None.
        """
        self._conts = conts
        self._persist_path = persist_path

    @property
    def name(self) -> str:
        """プロバイダ名。

        Returns:
            str: プロバイダ名
        """
        return ", ".join([cont.provider_name for cont in self._conts.values()])

    @property
    def persist_path(self) -> Optional[str]:
        """永続化パスを取得する。

        Returns:
            Optional[str]: 永続化パス
        """
        return self._persist_path

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
