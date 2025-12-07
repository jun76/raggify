from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.config_manager import ConfigManager
from ..config.vector_store_config import VectorStoreConfig, VectorStoreProvider
from ..core.const import EXTRA_PKG_NOT_FOUND_MSG, PJNAME_ALIAS
from ..core.utils import sanitize_str
from ..document_store.document_store_manager import DocumentStoreManager
from ..llama_like.core.schema import Modality

if TYPE_CHECKING:
    from ..embed.embed_manager import EmbedManager
    from .vector_store_manager import VectorStoreContainer, VectorStoreManager

__all__ = ["create_vector_store_manager"]


def create_vector_store_manager(
    cfg: ConfigManager,
    embed: EmbedManager,
    docstore: DocumentStoreManager,
) -> VectorStoreManager:
    """Create an instance of the vector store manager.

    Args:
        cfg (ConfigManager): Configuration manager.
        embed (EmbedManager): Embedding manager.
        docstore (DocumentStoreManager): Document store manager.

    Raises:
        RuntimeError: Failed to create an instance or provider not specified.

    Returns:
        VectorStoreManager: Vector store manager.
    """
    from .vector_store_manager import VectorStoreManager

    try:
        conts: dict[Modality, VectorStoreContainer] = {}
        if cfg.general.text_embed_provider:
            conts[Modality.TEXT] = _create_container(
                cfg=cfg,
                space_key=embed.space_key_text,
                dim=embed.get_container(Modality.TEXT).dim,
            )

        if cfg.general.image_embed_provider:
            conts[Modality.IMAGE] = _create_container(
                cfg=cfg,
                space_key=embed.space_key_image,
                dim=embed.get_container(Modality.IMAGE).dim,
            )

        if cfg.general.audio_embed_provider:
            conts[Modality.AUDIO] = _create_container(
                cfg=cfg,
                space_key=embed.space_key_audio,
                dim=embed.get_container(Modality.AUDIO).dim,
            )

        if cfg.general.video_embed_provider:
            conts[Modality.VIDEO] = _create_container(
                cfg=cfg,
                space_key=embed.space_key_video,
                dim=embed.get_container(Modality.VIDEO).dim,
            )
    except Exception as e:
        raise RuntimeError(f"failed to create vector store: {e}") from e

    if not conts:
        raise RuntimeError("no embedding providers are specified")

    return VectorStoreManager(conts=conts, embed=embed, docstore=docstore)


def _create_container(
    cfg: ConfigManager, space_key: str, dim: int
) -> VectorStoreContainer:
    """Create a container for each space key.

    Args:
        cfg (ConfigManager): Configuration manager.
        space_key (str): Space key.
        dim (int): Embedding dimension.

    Raises:
        RuntimeError: Unsupported provider.

    Returns:
        VectorStoreContainer: Container instance.
    """
    table_name = _generate_table_name(cfg, space_key)
    match cfg.general.vector_store_provider:
        case VectorStoreProvider.PGVECTOR:
            cont = _pgvector(cfg=cfg.vector_store, table_name=table_name, dim=dim)
        case VectorStoreProvider.CHROMA:
            cont = _chroma(cfg=cfg.vector_store, table_name=table_name)
        case VectorStoreProvider.REDIS:
            cont = _redis(cfg=cfg.vector_store, table_name=table_name, dim=dim)
        case _:
            raise RuntimeError(
                f"unsupported vector store: {cfg.general.vector_store_provider}"
            )

    return cont


def _generate_table_name(cfg: ConfigManager, space_key: str) -> str:
    """Generate a table name.

    Args:
        cfg (ConfigManager): Configuration manager.
        space_key (str): Space key.

    Raises:
        ValueError: Table name is too long.

    Returns:
        str: Table name.
    """
    return sanitize_str(
        f"{PJNAME_ALIAS}_{cfg.general.knowledgebase_name}_{space_key}_vec"
    )


# Container generators for each provider.
def _pgvector(
    cfg: VectorStoreConfig, table_name: str, dim: int
) -> VectorStoreContainer:
    try:
        from llama_index.vector_stores.postgres import PGVectorStore  # type: ignore
    except ImportError:
        raise ImportError(
            EXTRA_PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-vector-stores-postgres",
                extra="postgres",
                feature="PGVectorStore",
            )
        )

    from .vector_store_manager import VectorStoreContainer

    sec = cfg.pgvector_password
    if sec is None:
        raise ValueError("pgvector_password must be specified")

    return VectorStoreContainer(
        provider_name=VectorStoreProvider.PGVECTOR,
        store=PGVectorStore.from_params(
            host=cfg.pgvector_host,
            port=str(cfg.pgvector_port),
            database=cfg.pgvector_database,
            user=cfg.pgvector_user,
            password=sec,
            table_name=table_name,
            embed_dim=dim,
        ),
        table_name=table_name,
    )


def _chroma(cfg: VectorStoreConfig, table_name: str) -> VectorStoreContainer:
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore

    from .vector_store_manager import VectorStoreContainer

    if cfg.chroma_host is not None and cfg.chroma_port is not None:
        client = chromadb.HttpClient(
            host=cfg.chroma_host,
            port=cfg.chroma_port,
        )
    elif cfg.chroma_persist_dir:
        client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
    else:
        raise RuntimeError("persist_directory or host + port must be specified")

    collection = client.get_or_create_collection(table_name)

    return VectorStoreContainer(
        provider_name=VectorStoreProvider.CHROMA,
        store=ChromaVectorStore(chroma_collection=collection),
        table_name=table_name,
    )


def _redis(cfg: VectorStoreConfig, table_name: str, dim: int) -> VectorStoreContainer:
    try:
        from llama_index.vector_stores.redis import RedisVectorStore  # type: ignore
        from redisvl.schema import IndexSchema  # type: ignore
    except ImportError:
        raise ImportError(
            EXTRA_PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-vector-stores-redis",
                extra="redis",
                feature="RedisVectorStore",
            )
        )

    from .vector_store_manager import VectorStoreContainer

    schema = IndexSchema.from_dict(
        {
            "index": {"name": table_name},
            "fields": [
                {"name": "id", "type": "tag"},
                {"name": "doc_id", "type": "tag"},
                {"name": "text", "type": "text"},
                {
                    "name": "vector",
                    "type": "vector",
                    "attrs": {
                        "dims": dim,
                        "algorithm": "hnsw",
                        "distance_metric": "cosine",
                    },
                },
            ],
        }
    )

    return VectorStoreContainer(
        provider_name=VectorStoreProvider.REDIS,
        store=RedisVectorStore(
            redis_url=f"redis://{cfg.redis_host}:{cfg.redis_port}", schema=schema
        ),
        table_name=table_name,
    )
