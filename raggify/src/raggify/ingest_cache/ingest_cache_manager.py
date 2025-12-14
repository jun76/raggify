from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from ..llama_like.core.schema import Modality
from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.ingestion import IngestionCache
    from llama_index.core.schema import BaseNode, TransformComponent

__all__ = ["IngestCacheContainer", "IngestCacheManager"]


@dataclass(kw_only=True)
class IngestCacheContainer:
    """Container for ingest cache parameters per modality."""

    provider_name: str
    cache: IngestionCache
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

    def delete_nodes(
        self,
        modality: Modality,
        nodes: Sequence[BaseNode],
        transformations: Sequence[TransformComponent],
        persist_dir: Optional[Path],
    ) -> None:
        """Delete cache entries for given nodes and transformations.

        Args:
            modality (Modality): Modality.
            nodes (Sequence[BaseNode]): Nodes.
            transformations (Sequence[TransformComponent]): Transformations.
            persist_dir (Optional[Path]): Persist directory.

        Notes:
            Nodes must be the same (each transformations)
            as those used to create the cache entries.
        """
        from llama_index.core.ingestion.cache import DEFAULT_CACHE_NAME
        from llama_index.core.ingestion.pipeline import get_transformation_hash

        cache = self.get_container(modality).cache
        if cache is None:
            return

        for transform in transformations:
            try:
                key = get_transformation_hash(nodes, transform)
                cache.cache.delete(key=key, collection=cache.collection)
            except Exception as e:
                logger.warning(f"failed to delete cache entry: {e}")

        try:
            if persist_dir is not None:
                cache.persist(str(persist_dir / DEFAULT_CACHE_NAME))
        except Exception as e:
            logger.warning(f"failed to persist cache after delete: {e}")

    def delete_all(self, persist_dir: Optional[Path]) -> None:
        """Delete all caches.

        Args:
            persist_dir (Optional[Path]): Persist directory.
        """
        from llama_index.core.ingestion.cache import DEFAULT_CACHE_NAME

        for mod in self.modality:
            cache = self.get_container(mod).cache
            if cache is None:
                continue

            try:
                cache.clear()
                if persist_dir is not None:
                    cache.persist(str(persist_dir / DEFAULT_CACHE_NAME))
            except Exception as e:
                logger.warning(f"failed to clear {mod} cache, skipped: {e}")
                continue

        logger.info("all caches are deleted from cache store")
