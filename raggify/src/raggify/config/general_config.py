from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .default_settings import (
    DefaultSettings,
    EmbedProvider,
    RerankProvider,
    VectorStoreProvider,
)


@dataclass(kw_only=True)
class GeneralConfig:
    project_name: str = DefaultSettings.PROJECT_NAME
    version: str = DefaultSettings.VERSION
    knowledgebase_name: str = DefaultSettings.KNOWLEDGEBASE_NAME
    host: str = DefaultSettings.HOST
    port: int = DefaultSettings.PORT
    vector_store_provider: VectorStoreProvider = DefaultSettings.VECTOR_STORE_PROVIDER
    text_embed_provider: EmbedProvider = DefaultSettings.TEXT_EMBED_PROVIDER
    image_embed_provider: Optional[EmbedProvider] = DefaultSettings.IMAGE_EMBED_PROVIDER
    audio_embed_provider: Optional[EmbedProvider] = DefaultSettings.AUDIO_EMBED_PROVIDER
    rerank_provider: RerankProvider = DefaultSettings.RERANK_PROVIDER
    openai_base_url: Optional[str] = DefaultSettings.OPENAI_BASE_URL
    device: Literal["cpu", "cuda", "mps"] = DefaultSettings.DEVICE
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = (
        DefaultSettings.LOG_LEVEL
    )
