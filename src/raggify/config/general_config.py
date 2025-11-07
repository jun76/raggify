from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from ..core.const import DEFAULT_KNOWLEDGEBASE_NAME, DEFAULT_WORKSPACE_PATH
from .document_store_config import DocumentStoreProvider
from .embed_config import EmbedProvider
from .ingest_cache_config import IngestCacheStoreProvider
from .rerank_config import RerankProvider
from .vector_store_config import VectorStoreProvider


@dataclass(kw_only=True)
class GeneralConfig:
    workspace_path: Path = DEFAULT_WORKSPACE_PATH
    knowledgebase_name: str = DEFAULT_KNOWLEDGEBASE_NAME
    host: str = "localhost"
    port: int = 8000
    mcp: bool = False
    vector_store_provider: VectorStoreProvider = VectorStoreProvider.CHROMA
    document_store_provider: DocumentStoreProvider = DocumentStoreProvider.LOCAL
    ingest_cache_provider: IngestCacheStoreProvider = IngestCacheStoreProvider.LOCAL
    text_embed_provider: Optional[EmbedProvider] = EmbedProvider.OPENAI
    image_embed_provider: Optional[EmbedProvider] = EmbedProvider.VOYAGE
    audio_embed_provider: Optional[EmbedProvider] = None
    rerank_provider: Optional[RerankProvider] = None
    openai_base_url: Optional[str] = None
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
