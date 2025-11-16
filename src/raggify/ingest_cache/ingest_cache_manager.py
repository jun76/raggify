from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..embed.embed_manager import Modality
from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.ingestion import IngestionCache


@dataclass(kw_only=True)
class IngestCacheContainer:
    """モダリティ毎のインジェストキャッシュ関連パラメータを集約"""

    provider_name: str
    cache: Optional[IngestionCache]
    table_name: str


class IngestCacheManager:
    """インジェストキャッシュの管理クラス。"""

    def __init__(self, conts: dict[Modality, IngestCacheContainer]) -> None:
        """コンストラクタ

        Args:
            conts (dict[Modality, IngestCacheContainer]): インジェストキャッシュコンテナの辞書
        """
        self._conts = conts

        for modality, cont in conts.items():
            logger.debug(f"{cont.provider_name} {modality} ingest cache created")

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

    def get_container(self, modality: Modality) -> IngestCacheContainer:
        """モダリティ別のインジェストキャッシュコンテナを取得する。

        Args:
            modality (Modality): モダリティ

        Raises:
            RuntimeError: 未初期化

        Returns:
            IngestCacheContainer: インジェストキャッシュコンテナ
        """
        cont = self._conts.get(modality)
        if cont is None:
            raise RuntimeError(f"{modality} cache is not initialized")

        return cont

    def delete_all(self, persist_path: Optional[str]) -> None:
        """キャッシュを全削除する。

        Args:
            persist_path (Optional[str]): 永続化ディレクトリ
        """
        for mod in self.modality:
            cache = self.get_container(mod).cache
            if cache is None:
                continue

            try:
                cache.clear()
                if persist_path is not None:
                    cache.persist(persist_path)
            except Exception as e:
                logger.warning(f"failed to clear: {e}")
                return

        logger.info("all caches are deleted from cache store")
