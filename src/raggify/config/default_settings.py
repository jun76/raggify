from __future__ import annotations

from enum import StrEnum, auto
from typing import Any, Literal, Optional

from dotenv import load_dotenv

load_dotenv()


class VectorStoreProvider(StrEnum):
    CHROMA = auto()
    PGVECTOR = auto()
    REDIS = auto()


class DocumentStoreProvider(StrEnum):
    REDIS = auto()
    LOCAL = auto()


class IngestCacheStoreProvider(StrEnum):
    REDIS = auto()
    LOCAL = auto()


class EmbedProvider(StrEnum):
    CLIP = auto()
    OPENAI = auto()
    COHERE = auto()
    HUGGINGFACE = auto()
    CLAP = auto()
    VOYAGE = auto()


class RerankProvider(StrEnum):
    FLAGEMBEDDING = auto()
    COHERE = auto()


class EmbedModel(StrEnum):
    NAME = auto()
    DIM = auto()


class DefaultSettings:
    """各種設定値のデフォルト値管理クラス

    API キーやパスワード等は予め .env ファイルに記述しておく。
    """

    ##### Meta
    PROJECT_NAME: str = "raggify"
    VERSION: str = "0.1.0"
    USER_CONFIG_PATH: str = f"/etc/{PROJECT_NAME}/config.yaml"

    ##### General
    KNOWLEDGEBASE_NAME: str = "default"
    HOST: str = "localhost"
    PORT: int = 8000
    MCP: bool = False
    VECTOR_STORE_PROVIDER: VectorStoreProvider = VectorStoreProvider.CHROMA
    DOCUMENT_STORE_PROVIDER: DocumentStoreProvider = DocumentStoreProvider.LOCAL
    INGEST_CACHE_STORE_PROVIDER: IngestCacheStoreProvider = (
        IngestCacheStoreProvider.LOCAL
    )
    TEXT_EMBED_PROVIDER: Optional[EmbedProvider] = EmbedProvider.OPENAI
    IMAGE_EMBED_PROVIDER: Optional[EmbedProvider] = EmbedProvider.VOYAGE
    AUDIO_EMBED_PROVIDER: Optional[EmbedProvider] = None
    RERANK_PROVIDER: Optional[RerankProvider] = None
    OPENAI_BASE_URL: Optional[str] = None
    DEVICE: Literal["cpu", "cuda", "mps"] = "cpu"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "DEBUG"

    ##### Vector Store
    # General
    CACHE_LOAD_LIMIT: int = 10000
    CHECK_UPDATE: bool = False

    # Chroma
    CHROMA_PERSIST_DIR: str = f"/etc/{PROJECT_NAME}/{PROJECT_NAME}_db"
    CHROMA_HOST: Optional[str] = None
    CHROMA_PORT: Optional[int] = None
    CHROMA_TENANT: Optional[str] = None
    CHROMA_DATABASE: Optional[str] = None

    # PGVector
    PGVECTOR_HOST: str = "localhost"
    PGVECTOR_PORT: int = 5432
    PGVECTOR_DATABASE: str = PROJECT_NAME
    PGVECTOR_USER: str = PROJECT_NAME
    PGVECTOR_PASSWORD: Optional[str] = None

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    ##### Document Store
    # Redis
    # same as document store settings

    ##### Ingest Cache Store
    # Redis
    # same as document store settings

    ##### Meta Store
    META_STORE_PATH: str = f"/etc/{PROJECT_NAME}/{PROJECT_NAME}_metas.db"

    ##### Embedding
    # Text
    OPENAI_EMBED_MODEL_TEXT: dict[str, Any] = {
        EmbedModel.NAME.value: "text-embedding-3-small",
        EmbedModel.DIM.value: 1536,
    }
    COHERE_EMBED_MODEL_TEXT: dict[str, Any] = {
        EmbedModel.NAME.value: "embed-v4.0",
        EmbedModel.DIM.value: 1536,
    }
    CLIP_EMBED_MODEL_TEXT: dict[str, Any] = {
        EmbedModel.NAME.value: "ViT-B/32",
        EmbedModel.DIM.value: 512,
    }
    HUGGINGFACE_EMBED_MODEL_TEXT: dict[str, Any] = {
        EmbedModel.NAME.value: "intfloat/multilingual-e5-base",
        EmbedModel.DIM.value: 768,
    }
    VOYAGE_EMBED_MODEL_TEXT: dict[str, Any] = {
        EmbedModel.NAME.value: "voyage-3.5",
        EmbedModel.DIM.value: 2048,
    }

    # Image
    COHERE_EMBED_MODEL_IMAGE: dict[str, Any] = {
        EmbedModel.NAME.value: "embed-v4.0",
        EmbedModel.DIM.value: 1536,
    }
    CLIP_EMBED_MODEL_IMAGE: dict[str, Any] = {
        EmbedModel.NAME.value: "ViT-B/32",
        EmbedModel.DIM.value: 512,
    }
    HUGGINGFACE_EMBED_MODEL_IMAGE: dict[str, Any] = {
        EmbedModel.NAME.value: "llamaindex/vdr-2b-multi-v1",
        EmbedModel.DIM.value: 1536,
    }
    VOYAGE_EMBED_MODEL_IMAGE: dict[str, Any] = {
        EmbedModel.NAME.value: "voyage-multimodal-3",
        EmbedModel.DIM.value: 1024,
    }

    # Audio
    CLAP_EMBED_MODEL_AUDIO: dict[str, Any] = {
        EmbedModel.NAME.value: "effect_varlen",
        EmbedModel.DIM.value: 512,
    }

    ##### Ingest
    # General
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    UPLOAD_DIR: str = f"/etc/{PROJECT_NAME}/upload"
    PIPE_PERSIST_DIR: str = f"/etc/{PROJECT_NAME}/pipeline_storage"
    # Web
    USER_AGENT: str = PROJECT_NAME
    LOAD_ASSET: bool = True
    REQ_PER_SEC: int = 2
    TIMEOUT_SEC: int = 30
    SAME_ORIGIN: bool = True

    ##### Rerank
    FLAGEMBEDDING_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    COHERE_RERANK_MODEL: str = "rerank-multilingual-v3.0"
    TOPK: int = 10
