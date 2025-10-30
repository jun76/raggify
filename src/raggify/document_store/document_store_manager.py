from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..embed.embed_manager import Modality

if TYPE_CHECKING:
    from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore


@dataclass
class DocumentStoreContainer:
    """モダリティ毎のドキュメントストア関連パラメータを集約"""

    provider_name: str
    store: KVDocumentStore
    table_name: str


class DocumentStoreManager:
    """ドキュメントストアの管理クラス。"""

    def __init__(
        self,
        conts: dict[Modality, DocumentStoreContainer],
    ) -> None:
        """コンストラクタ

        Args:
            conts (dict[Modality, DocumentStoreContainer]): ドキュメントストアコンテナの辞書
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
        """このドキュメントストアがサポートするモダリティ一覧。

        Returns:
            set[Modality]: モダリティ一覧
        """
        return set(self._conts.keys())

    def get_container(self, modality: Modality) -> DocumentStoreContainer:
        """モダリティ別のドキュメントストアコンテナを取得する。

        Args:
            modality (Modality): モダリティ

        Raises:
            RuntimeError: 未初期化

        Returns:
            DocumentStoreContainer: ドキュメントストアコンテナ
        """
        cont = self._conts.get(modality)
        if cont is None:
            raise RuntimeError(f"store {modality} is not initialized")

        return cont
