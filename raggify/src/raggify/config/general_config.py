from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from mashumaro import DataClassDictMixin

from ..core.const import DEFAULT_KNOWLEDGEBASE_NAME
from .document_store_config import DocumentStoreProvider
from .embed_config import EmbedProvider
from .ingest_cache_config import IngestCacheProvider
from .rerank_config import RerankProvider
from .vector_store_config import VectorStoreProvider


@dataclass(kw_only=True)
class GeneralConfig(DataClassDictMixin):
    knowledgebase_name: str = DEFAULT_KNOWLEDGEBASE_NAME
    host: str = "localhost"
    port: int = 8000
    mcp: bool = False
    vector_store_provider: VectorStoreProvider = VectorStoreProvider.CHROMA
    document_store_provider: DocumentStoreProvider = DocumentStoreProvider.LOCAL
    ingest_cache_provider: IngestCacheProvider = IngestCacheProvider.LOCAL
    text_embed_provider: Optional[EmbedProvider] = EmbedProvider.OPENAI
    image_embed_provider: Optional[EmbedProvider] = EmbedProvider.COHERE
    audio_embed_provider: Optional[EmbedProvider] = EmbedProvider.BEDROCK
    video_embed_provider: Optional[EmbedProvider] = EmbedProvider.BEDROCK
    use_modality_fallback: bool = False
    rerank_provider: Optional[RerankProvider] = RerankProvider.COHERE
    openai_base_url: Optional[str] = None
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
