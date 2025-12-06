from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from mashumaro import DataClassDictMixin

from ..core.const import DEFAULT_KNOWLEDGEBASE_NAME
from .document_store_config import DocumentStoreProvider
from .embed_config import EmbedProvider
from .ingest_cache_config import IngestCacheProvider
from .ingest_config import ParserProvider
from .llm_config import LLMProvider
from .rerank_config import RerankProvider
from .vector_store_config import VectorStoreProvider

__all__ = ["GeneralConfig"]


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
    image_embed_provider: Optional[EmbedProvider] = None
    audio_embed_provider: Optional[EmbedProvider] = None
    video_embed_provider: Optional[EmbedProvider] = None
    text_summarize_transform_provider: Optional[LLMProvider] = LLMProvider.OPENAI
    image_summarize_transform_provider: Optional[LLMProvider] = None
    audio_summarize_transform_provider: Optional[LLMProvider] = None
    video_summarize_transform_provider: Optional[LLMProvider] = None
    rerank_provider: Optional[RerankProvider] = None
    parser_provider: Optional[ParserProvider] = ParserProvider.LOCAL
    use_modality_fallback: bool = False
    openai_base_url: Optional[str] = None
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"
