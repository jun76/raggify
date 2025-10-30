from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ..config.config_manager import ConfigManager
from ..config.default_settings import EmbedModel as EM
from ..config.default_settings import EmbedProvider
from ..llama.core.schema import Modality
from ..logger import logger

if TYPE_CHECKING:
    from .embed_manager import EmbedContainer, EmbedManager

__all__ = ["create_embed_manager"]


def create_embed_manager(cfg: ConfigManager) -> EmbedManager:
    """埋め込み管理のインスタンスを生成する。

    Args:
        cfg (ConfigManager): 設定管理

    Raises:
        RuntimeError: インスタンス生成に失敗またはプロバイダ指定漏れ

    Returns:
        EmbedManager: 埋め込み管理
    """
    from .embed_manager import EmbedManager

    try:
        conts: dict[Modality, EmbedContainer] = {}
        if cfg.general.text_embed_provider:
            match cfg.general.text_embed_provider:
                case EmbedProvider.OPENAI:
                    cont = _openai_text(cfg)
                case EmbedProvider.COHERE:
                    cont = _cohere_text(cfg)
                case EmbedProvider.CLIP:
                    cont = _clip_text(cfg)
                case EmbedProvider.HUGGINGFACE:
                    cont = _huggingface_text(cfg)
                case EmbedProvider.VOYAGE:
                    cont = _voyage_text(cfg)
                case _:
                    raise ValueError(
                        "unsupported text embed provider: "
                        f"{cfg.general.text_embed_provider}"
                    )
            conts[Modality.TEXT] = cont

        if cfg.general.image_embed_provider:
            match cfg.general.image_embed_provider:
                case EmbedProvider.COHERE:
                    cont = _cohere_image(cfg)
                case EmbedProvider.CLIP:
                    cont = _clip_image(cfg)
                case EmbedProvider.HUGGINGFACE:
                    cont = _huggingface_image(cfg)
                case EmbedProvider.VOYAGE:
                    cont = _voyage_image(cfg)
                case _:
                    raise ValueError(
                        "unsupported image embed provider: "
                        f"{cfg.general.image_embed_provider}"
                    )
            conts[Modality.IMAGE] = cont

        if cfg.general.audio_embed_provider:
            match cfg.general.audio_embed_provider:
                case EmbedProvider.CLAP:
                    cont = _clap_audio(cfg)
                case _:
                    raise ValueError(
                        "unsupported audio embed provider: "
                        f"{cfg.general.audio_embed_provider}"
                    )
            conts[Modality.AUDIO] = cont
    except (ValidationError, ValueError) as e:
        raise RuntimeError("invalid settings") from e
    except Exception as e:
        raise RuntimeError("failed to create embedding") from e

    if not conts:
        raise RuntimeError("no embedding providers are specified")

    return EmbedManager(conts)


# 以下、プロバイダ毎のコンテナ生成ヘルパー
def _openai_text(cfg: ConfigManager) -> EmbedContainer:
    from llama_index.embeddings.openai.base import OpenAIEmbedding

    from .embed_manager import EmbedContainer

    dim = cfg.embed.openai_embed_model_text[EM.DIM]

    return EmbedContainer(
        provider_name=EmbedProvider.OPENAI,
        embed=OpenAIEmbedding(
            model=cfg.embed.openai_embed_model_text[EM.NAME],
            api_base=cfg.general.openai_base_url,
            dimensions=dim,
        ),
        dim=dim,
    )


def _cohere_text(cfg: ConfigManager) -> EmbedContainer:
    from llama_index.embeddings.cohere.base import CohereEmbedding

    from .embed_manager import EmbedContainer

    return EmbedContainer(
        provider_name=EmbedProvider.COHERE,
        embed=CohereEmbedding(
            api_key=os.getenv("COHERE_API_KEY"),
            model_name=cfg.embed.cohere_embed_model_text[EM.NAME],
        ),
        dim=cfg.embed.cohere_embed_model_text[EM.DIM],
    )


def _cohere_image(cfg: ConfigManager) -> EmbedContainer:
    from llama_index.embeddings.cohere.base import CohereEmbedding

    from .embed_manager import EmbedContainer

    return EmbedContainer(
        provider_name=EmbedProvider.COHERE,
        embed=CohereEmbedding(
            api_key=os.getenv("COHERE_API_KEY"),
            model_name=cfg.embed.cohere_embed_model_image[EM.NAME],
        ),
        dim=cfg.embed.cohere_embed_model_image[EM.DIM],
    )


def _clip_text(cfg: ConfigManager) -> EmbedContainer:
    try:
        from llama_index.embeddings.clip import ClipEmbedding
    except ImportError as e:
        logger.error(
            "llama-index-embeddings-clip not found. Try install with [local] option."
        )
        raise e

    from .embed_manager import EmbedContainer

    return EmbedContainer(
        provider_name=EmbedProvider.CLIP,
        embed=ClipEmbedding(
            model_name=cfg.embed.clip_embed_model_text[EM.NAME],
            device=cfg.general.device,
        ),
        dim=cfg.embed.clip_embed_model_text[EM.DIM],
    )


def _clip_image(cfg: ConfigManager) -> EmbedContainer:
    try:
        from llama_index.embeddings.clip import ClipEmbedding
    except ImportError as e:
        logger.error(
            "llama-index-embeddings-clip not found. Try install with [local] option."
        )
        raise e

    from .embed_manager import EmbedContainer

    return EmbedContainer(
        provider_name=EmbedProvider.CLIP,
        embed=ClipEmbedding(
            model_name=cfg.embed.clip_embed_model_image[EM.NAME],
            device=cfg.general.device,
        ),
        dim=cfg.embed.clip_embed_model_image[EM.DIM],
    )


def _huggingface_text(cfg: ConfigManager) -> EmbedContainer:
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as e:
        logger.error(
            "llama-index-embeddings-huggingface not found. Try install with [local] option."
        )
        raise e

    from .embed_manager import EmbedContainer

    return EmbedContainer(
        provider_name=EmbedProvider.HUGGINGFACE,
        embed=HuggingFaceEmbedding(
            model_name=cfg.embed.huggingface_embed_model_text[EM.NAME],
            device=cfg.general.device,
        ),
        dim=cfg.embed.huggingface_embed_model_text[EM.DIM],
    )


def _huggingface_image(cfg: ConfigManager) -> EmbedContainer:
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as e:
        logger.error(
            "llama-index-embeddings-huggingface not found. Try install with [local] option."
        )
        raise e

    from .embed_manager import EmbedContainer

    return EmbedContainer(
        provider_name=EmbedProvider.HUGGINGFACE,
        embed=HuggingFaceEmbedding(
            model_name=cfg.embed.huggingface_embed_model_image[EM.NAME],
            device=cfg.general.device,
            trust_remote_code=True,
        ),
        dim=cfg.embed.huggingface_embed_model_image[EM.DIM],
    )


def _clap_audio(cfg: ConfigManager) -> EmbedContainer:
    try:
        from ..llama.embeddings.clap import ClapEmbedding
    except ImportError as e:
        logger.error("laion-clap not found. Try install with [local] option.")
        raise e

    from .embed_manager import EmbedContainer

    return EmbedContainer(
        provider_name=EmbedProvider.CLAP,
        embed=ClapEmbedding(
            model_name=cfg.embed.clap_embed_model_audio[EM.NAME],
            device=cfg.general.device,
        ),
        dim=cfg.embed.clap_embed_model_audio[EM.DIM],
    )


def _voyage_text(cfg: ConfigManager) -> EmbedContainer:
    from llama_index.embeddings.voyageai.base import VoyageEmbedding

    from .embed_manager import EmbedContainer

    dim = cfg.embed.voyage_embed_model_text[EM.DIM]

    return EmbedContainer(
        provider_name=EmbedProvider.VOYAGE,
        embed=VoyageEmbedding(
            api_key=os.getenv("VOYAGE_API_KEY"),
            model_name=cfg.embed.voyage_embed_model_text[EM.NAME],
            truncation=False,
            output_dimension=dim,
        ),
        dim=dim,
    )


def _voyage_image(cfg: ConfigManager) -> EmbedContainer:
    from llama_index.embeddings.voyageai.base import VoyageEmbedding

    from .embed_manager import EmbedContainer

    dim = cfg.embed.voyage_embed_model_image[EM.DIM]

    return EmbedContainer(
        provider_name=EmbedProvider.VOYAGE,
        embed=VoyageEmbedding(
            api_key=os.getenv("VOYAGE_API_KEY"),
            model_name=cfg.embed.voyage_embed_model_image[EM.NAME],
            truncation=False,
            output_dimension=dim,
        ),
        dim=dim,
    )
