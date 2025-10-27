from __future__ import annotations

from enum import StrEnum, auto
from typing import Any, Literal, Optional

from dotenv import load_dotenv

load_dotenv()


class VectorStoreProvider(StrEnum):
    CHROMA = auto()
    PGVECTOR = auto()


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


class ModelKey:
    NAME = "name"
    DIM = "dim"


class DefaultSettings:
    """各種設定値のデフォルト値管理クラス

    API キーやパスワード等は予め .env ファイルに記述しておく。
    """

    ##### Meta
    PROJECT_NAME: str = "raggify"
    VERSION: str = "1.0"
    USER_CONFIG_PATH: str = f"/etc/{PROJECT_NAME}/config.yaml"

    ##### General
    KNOWLEDGEBASE_NAME: str = "default"
    HOST: str = "localhost"
    PORT: int = 8000
    MCP: bool = False
    VECTOR_STORE_PROVIDER: VectorStoreProvider = VectorStoreProvider.CHROMA
    TEXT_EMBED_PROVIDER: Optional[EmbedProvider] = EmbedProvider.HUGGINGFACE
    IMAGE_EMBED_PROVIDER: Optional[EmbedProvider] = EmbedProvider.CLIP
    AUDIO_EMBED_PROVIDER: Optional[EmbedProvider] = EmbedProvider.CLAP
    RERANK_PROVIDER: RerankProvider = RerankProvider.FLAGEMBEDDING
    OPENAI_BASE_URL: Optional[str] = None
    DEVICE: Literal["cpu", "cuda", "mps"] = "cuda"
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

    ##### Meta Store
    META_STORE_PATH: str = f"/etc/{PROJECT_NAME}/{PROJECT_NAME}_metas.db"

    ##### Embedding
    # Text
    OPENAI_EMBED_MODEL_TEXT: dict[str, Any] = {
        ModelKey.NAME: "text-embedding-3-small",
        ModelKey.DIM: 1536,
    }
    COHERE_EMBED_MODEL_TEXT: dict[str, Any] = {
        ModelKey.NAME: "embed-v4.0",
        ModelKey.DIM: 1536,
    }
    CLIP_EMBED_MODEL_TEXT: dict[str, Any] = {
        ModelKey.NAME: "ViT-B/32",
        ModelKey.DIM: 512,
    }
    HUGGINGFACE_EMBED_MODEL_TEXT: dict[str, Any] = {
        ModelKey.NAME: "intfloat/multilingual-e5-base",
        ModelKey.DIM: 768,
    }
    VOYAGE_EMBED_MODEL_TEXT: dict[str, Any] = {
        ModelKey.NAME: "voyage-3.5",
        ModelKey.DIM: 2048,
    }

    # Image
    COHERE_EMBED_MODEL_IMAGE: dict[str, Any] = {
        ModelKey.NAME: "embed-v4.0",
        ModelKey.DIM: 1536,
    }
    CLIP_EMBED_MODEL_IMAGE: dict[str, Any] = {
        ModelKey.NAME: "ViT-B/32",
        ModelKey.DIM: 512,
    }
    HUGGINGFACE_EMBED_MODEL_IMAGE: dict[str, Any] = {
        ModelKey.NAME: "llamaindex/vdr-2b-multi-v1",
        ModelKey.DIM: 1536,
    }
    VOYAGE_EMBED_MODEL_IMAGE: dict[str, Any] = {
        ModelKey.NAME: "voyage-multimodal-3",
        ModelKey.DIM: 1024,
    }

    # Audio
    CLAP_EMBED_MODEL_AUDIO: dict[str, Any] = {
        ModelKey.NAME: "effect_varlen",
        ModelKey.DIM: 512,
    }

    ##### Ingest
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    USER_AGENT: str = PROJECT_NAME
    UPLOAD_DIR: str = "upload"

    ##### Rerank
    FLAGEMBEDDING_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    COHERE_RERANK_MODEL: str = "rerank-multilingual-v3.0"
    TOPK: int = 10
