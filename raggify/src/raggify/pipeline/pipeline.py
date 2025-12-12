from __future__ import annotations

from typing import TYPE_CHECKING

from .pipeline_manager import PipelineManager

if TYPE_CHECKING:
    from ..config.config_manager import ConfigManager
    from ..document_store.document_store_manager import DocumentStoreManager
    from ..ingest_cache.ingest_cache_manager import IngestCacheManager
    from ..vector_store.vector_store_manager import VectorStoreManager


__all__ = ["create_pipeline_manager"]


def create_pipeline_manager(
    cfg: ConfigManager,
    vector_store: VectorStoreManager,
    ingest_cache: IngestCacheManager,
    document_store: DocumentStoreManager,
) -> PipelineManager:
    """Create a pipeline manager instance.

    Args:
        cfg (ConfigManager): Configuration manager.
        vector_store (VectorStoreManager): Vector store manager.
        ingest_cache (IngestCacheManager): Ingest cache manager.
        document_store (DocumentStoreManager): Document store manager.

    Returns:
        PipelineManager: Pipeline manager.
    """
    return PipelineManager(
        cfg=cfg,
        vector_store=vector_store,
        ingest_cache=ingest_cache,
        document_store=document_store,
    )
