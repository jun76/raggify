from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .document_store_config import DocumentStoreProvider
from .embed_config import EmbedProvider
from .ingest_cache_store_config import IngestCacheStoreProvider
from .rerank_config import RerankProvider
from .vector_store_config import VectorStoreProvider


@dataclass(kw_only=True)
class GeneralConfig:
    knowledgebase_name: str = "default"
    host: str = "localhost"
    port: int = 8000
    mcp: bool = False
    vector_store_provider: VectorStoreProvider = VectorStoreProvider.CHROMA
    document_store_provider: DocumentStoreProvider = DocumentStoreProvider.LOCAL
    ingest_cache_store_provider: IngestCacheStoreProvider = (
        IngestCacheStoreProvider.LOCAL
    )
    text_embed_provider: Optional[EmbedProvider] = EmbedProvider.OPENAI
    image_embed_provider: Optional[EmbedProvider] = EmbedProvider.VOYAGE
    audio_embed_provider: Optional[EmbedProvider] = None
    rerank_provider: Optional[RerankProvider] = None
    openai_base_url: Optional[str] = None
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
