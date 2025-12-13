from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.ingestion.pipeline import DocstoreStrategy

from ..ingest.transform.base_transform import BaseTransform
from ..logger import logger

if TYPE_CHECKING:
    from llama_index.core.ingestion.cache import IngestionCache
    from llama_index.core.schema import BaseNode, TransformComponent
    from llama_index.core.storage.docstore import BaseDocumentStore
    from llama_index.core.vector_stores.types import BasePydanticVectorStore

    from ..config.config_manager import ConfigManager
    from ..document_store.document_store_manager import DocumentStoreManager
    from ..ingest_cache.ingest_cache_manager import IngestCacheManager
    from ..llama_like.core.schema import Modality
    from ..vector_store.vector_store_manager import VectorStoreManager


__all__ = ["TracablePipeline", "PipelineManager"]


class TracablePipeline(IngestionPipeline):
    """Manages transformed nodes for rollback in case of pipeline run failure."""

    def __init__(
        self,
        transformations: list[TransformComponent],
        vector_store: BasePydanticVectorStore,
        cache: IngestionCache,
        docstore: BaseDocumentStore,
        docstore_strategy: DocstoreStrategy,
        **kwargs,
    ) -> None:
        """Constructor.

        Args:
            transformations (list[TransformComponent]): List of transformation components.
            vector_store (BasePydanticVectorStore): Vector store instance.
            cache (IngestionCache): Ingestion cache instance.
            docstore (BaseDocumentStore): Document store instance.
            docstore_strategy (DocstoreStrategy): Document store strategy.
        """
        super().__init__(
            transformations=transformations,
            vector_store=vector_store,
            cache=cache,
            docstore=docstore,
            docstore_strategy=docstore_strategy,
            **kwargs,
        )
        self._transformed_nodes: list[tuple[TransformComponent, Sequence[BaseNode]]] = (
            []
        )

        # Set pipe callback for rollback tracking
        for transformation in transformations:
            if isinstance(transformation, BaseTransform):
                transformation.set_pipe_callback(self.record_nodes)

    @property
    def nodes(self) -> list[tuple[TransformComponent, Sequence[BaseNode]]]:
        return self._transformed_nodes

    def reset_nodes(self) -> None:
        """Reset transformed nodes."""
        self._transformed_nodes = []

    def record_nodes(
        self, transform: TransformComponent, nodes: Sequence[BaseNode]
    ) -> None:
        """Records transformed nodes.

        Args:
            transform (TransformComponent): Transform applied.
            nodes (Sequence[BaseNode]): Nodes after transformation.
        """
        self._transformed_nodes.append((transform, nodes))


class PipelineManager:
    """Manager class for ingestion pipelines."""

    def __init__(
        self,
        cfg: ConfigManager,
        vector_store: VectorStoreManager,
        ingest_cache: IngestCacheManager,
        document_store: DocumentStoreManager,
    ) -> None:
        """Constructor.

        Args:
            cfg (ConfigManager): Configuration manager.
            vector_store (VectorStoreManager): Vector store manager.
            ingest_cache (IngestCacheManager): Ingest cache manager.
            document_store (DocumentStoreManager): Document store manager.
        """
        self.cfg = cfg
        self.vector_store = vector_store
        self.ingest_cache = ingest_cache
        self.document_store = document_store
        self._pipeline_lock = threading.Lock()

    def _use_local_workspace(self) -> bool:
        """Whether to persist cache or document store locally.

        Returns:
            bool: True when persisting locally.
        """
        from ..config.document_store_config import DocumentStoreProvider
        from ..config.ingest_cache_config import IngestCacheProvider

        cfg = self.cfg.general
        if (cfg.ingest_cache_provider is IngestCacheProvider.LOCAL) or (
            cfg.document_store_provider is DocumentStoreProvider.LOCAL
        ):
            return True

        return False

    def build(
        self,
        modality: Modality,
        transformations: list[TransformComponent],
        persist_dir: Optional[Path] = None,
    ) -> TracablePipeline:
        """Create or load an ingestion pipeline.

        Args:
            modality (Modality): Modality.
            transformations (list[TransformComponent]): list of transforms.
            persist_dir (Optional[Path], optional): Persistence directory. Defaults to None.

        Returns:
            TracablePipeline: Pipeline instance.
        """
        pipe = TracablePipeline(
            transformations=transformations,
            vector_store=self.vector_store.get_container(modality).store,
            cache=self.ingest_cache.get_container(modality).cache,
            docstore=self.document_store.store,
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )

        if not self._use_local_workspace():
            return pipe

        if not (persist_dir and persist_dir.exists()):
            return pipe

        try:
            pipe.load(str(persist_dir))
            with self._pipeline_lock:
                self.ingest_cache.get_container(modality).cache = pipe.cache
                if pipe.docstore is None:
                    logger.warning("pipeline has no docstore")
                else:
                    self.document_store.store = pipe.docstore
                logger.debug(f"loaded pipeline from {persist_dir}")
        except Exception as e:
            logger.warning(f"failed to load persist dir: {e}")

        return pipe

    def persist(
        self,
        pipe: TracablePipeline,
        modality: Modality,
        persist_dir: Optional[Path] = None,
    ) -> None:
        """Persist the pipeline to storage.

        Args:
            pipe (TracablePipeline): Pipeline instance.
            modality (Modality): Modality.
            persist_dir (Optional[Path], optional): Persistence directory. Defaults to None.
        """
        if not self._use_local_workspace():
            return

        if persist_dir is None:
            logger.warning(f"persist dir not specified, skipped persisting")
            return

        try:
            pipe.persist(str(persist_dir))
            with self._pipeline_lock:
                self.ingest_cache.get_container(modality).cache = pipe.cache
                if pipe.docstore is None:
                    logger.warning("pipeline has no docstore")
                else:
                    self.document_store.store = pipe.docstore
                logger.debug(f"persisted pipeline to {persist_dir}")
        except Exception as e:
            logger.warning(f"failed to persist: {e}")

    def delete_all(self) -> None:
        """Delete all data persisted in each store."""
        with self._pipeline_lock:
            if self._use_local_workspace():
                persist_dir = self.cfg.pipeline.persist_dir
            else:
                persist_dir = None

            if not self.vector_store.delete_all():
                ref_doc_ids = self.document_store.get_ref_doc_ids()
                self.vector_store.delete_nodes(ref_doc_ids)

            self.ingest_cache.delete_all(persist_dir)
            self.document_store.delete_all(persist_dir)
