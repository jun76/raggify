from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..embed.embed_manager import Modality
from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.ingestion import IngestionCache


@dataclass(kw_only=True)
class IngestCacheContainer:
    """Container for ingest cache parameters per modality."""

    provider_name: str
    cache: Optional[IngestionCache]
    table_name: str


class IngestCacheManager:
    """Manager class for ingest caches."""

    def __init__(self, conts: dict[Modality, IngestCacheContainer]) -> None:
        """Constructor.

        Args:
            conts (dict[Modality, IngestCacheContainer]):
                Mapping of modality to ingest cache container.
        """
        self._conts = conts

        for modality, cont in conts.items():
            logger.debug(f"{cont.provider_name} {modality} ingest cache created")

    @property
    def name(self) -> str:
        """Provider names.

        Returns:
            str: Provider names.
        """
        return ", ".join([cont.provider_name for cont in self._conts.values()])

    @property
    def modality(self) -> set[Modality]:
        """Modalities supported by this ingest cache manager.

        Returns:
            set[Modality]: Modalities.
        """
        return set(self._conts.keys())

    def get_container(self, modality: Modality) -> IngestCacheContainer:
        """Get the ingest cache container for a modality.

        Args:
            modality (Modality): Modality.

        Raises:
            RuntimeError: If uninitialized.

        Returns:
            IngestCacheContainer: Ingest cache container.
        """
        cont = self._conts.get(modality)
        if cont is None:
            raise RuntimeError(f"{modality} cache is not initialized")

        return cont

    def delete_all(self, persist_path: Optional[str]) -> None:
        """Delete all caches.

        Args:
            persist_path (Optional[str]): Persist directory.
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
                logger.warning(f"failed to clear {mod} cache: {e}")
                return

        logger.info("all caches are deleted from cache store")
