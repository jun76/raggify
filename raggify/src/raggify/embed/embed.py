from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ..config.config_manager import ConfigManager
from ..config.embed_config import EmbedConfig
from ..config.embed_config import EmbedModel as EM
from ..config.embed_config import EmbedProvider
from ..core.const import EXTRA_PKG_NOT_FOUND_MSG
from ..llama_like.core.schema import Modality
from ..logger import logger

if TYPE_CHECKING:
    from .embed_manager import EmbedContainer, EmbedManager

__all__ = ["create_embed_manager"]


def create_embed_manager(cfg: ConfigManager) -> EmbedManager:
    """Create an embedding manager instance.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        RuntimeError: If instantiation fails or providers are not specified.

    Returns:
        EmbedManager: Embedding manager.
    """
    from .embed_manager import EmbedManager

    try:
        conts: dict[Modality, EmbedContainer] = {}
        if cfg.general.text_embed_provider:
            conts[Modality.TEXT] = _create_text_embed(cfg)
        if cfg.general.image_embed_provider:
            conts[Modality.IMAGE] = _create_image_embed(cfg)
        if cfg.general.audio_embed_provider:
            conts[Modality.AUDIO] = _create_audio_embed(cfg)
        if cfg.general.video_embed_provider:
            conts[Modality.VIDEO] = _create_video_embed(cfg)
    except (ValidationError, ValueError) as e:
        raise RuntimeError("invalid settings") from e
    except Exception as e:
        raise RuntimeError("failed to create embedding") from e

    if not conts:
        raise RuntimeError("no embedding providers are specified")

    return EmbedManager(
        conts=conts,
        embed_batch_size=cfg.embed.batch_size,
        batch_interval_sec=cfg.embed.batch_interval_sec,
    )


def _create_text_embed(cfg: ConfigManager) -> EmbedContainer:
    """Create text embed container.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        ValueError: If text embed provider is not specified or unsupported.

    Returns:
        EmbedContainer: Text embed container.
    """
    provider = cfg.general.text_embed_provider
    if provider is None:
        raise ValueError("text embed provider is not specified")
    match provider:
        case EmbedProvider.OPENAI:
            return _openai_text(cfg)
        case EmbedProvider.COHERE:
            return _cohere_text(cfg.embed)
        case EmbedProvider.CLIP:
            return _clip_text(cfg)
        case EmbedProvider.CLAP:
            return _clap_text(cfg)
        case EmbedProvider.HUGGINGFACE:
            return _huggingface_text(cfg)
        case EmbedProvider.VOYAGE:
            return _voyage_text(cfg.embed)
        case EmbedProvider.BEDROCK:
            return _bedrock_text(cfg.embed)
        case _:
            raise ValueError(f"unsupported text embed provider: {provider}")


def _create_image_embed(cfg: ConfigManager) -> EmbedContainer:
    """Create image embed container.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        ValueError: If image embed provider is not specified or unsupported.

    Returns:
        EmbedContainer: Image embed container.
    """
    provider = cfg.general.image_embed_provider
    if provider is None:
        raise ValueError("image embed provider is not specified")
    match provider:
        case EmbedProvider.COHERE:
            return _cohere_image(cfg.embed)
        case EmbedProvider.CLIP:
            return _clip_image(cfg)
        case EmbedProvider.HUGGINGFACE:
            return _huggingface_image(cfg)
        case EmbedProvider.VOYAGE:
            return _voyage_image(cfg.embed)
        case EmbedProvider.BEDROCK:
            return _bedrock_image(cfg.embed)
        case _:
            raise ValueError(f"unsupported image embed provider: {provider}")


def _create_audio_embed(cfg: ConfigManager) -> EmbedContainer:
    """Create audio embed container.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        ValueError: If audio embed provider is not specified or unsupported.

    Returns:
        EmbedContainer: Audio embed container.
    """
    provider = cfg.general.audio_embed_provider
    if provider is None:
        raise ValueError("audio embed provider is not specified")
    match provider:
        case EmbedProvider.CLAP:
            return _clap_audio(cfg)
        case EmbedProvider.BEDROCK:
            return _bedrock_audio(cfg.embed)
        case _:
            raise ValueError(f"unsupported audio embed provider: {provider}")


def _create_video_embed(cfg: ConfigManager) -> EmbedContainer:
    """Create video embed container.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        ValueError: If video embed provider is not specified or unsupported.

    Returns:
        EmbedContainer: Video embed container.
    """
    provider = cfg.general.video_embed_provider
    if provider is None:
        raise ValueError("video embed provider is not specified")
    match provider:
        case EmbedProvider.BEDROCK:
            return _bedrock_video(cfg.embed)
        case _:
            raise ValueError(f"unsupported video embed provider: {provider}")


# Container generation helpers per provider
def _openai_text(cfg: ConfigManager) -> EmbedContainer:
    from llama_index.embeddings.openai.base import OpenAIEmbedding

    from .embed_manager import EmbedContainer

    model = cfg.embed.openai_embed_model_text

    return EmbedContainer(
        provider_name=EmbedProvider.OPENAI,
        embed=OpenAIEmbedding(
            model=model[EM.NAME],
            api_base=cfg.general.openai_base_url,
            dimensions=model[EM.DIM],
        ),
        dim=model[EM.DIM],
        alias=model[EM.ALIAS],
    )


def _cohere(model: dict[str, Any], extra: str) -> EmbedContainer:
    try:
        from llama_index.embeddings.cohere.base import CohereEmbedding  # type: ignore
    except ImportError:
        raise ImportError(
            EXTRA_PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-embeddings-cohere",
                extra=extra,
                feature="CohereEmbedding",
            )
        )

    from .embed_manager import EmbedContainer

    return EmbedContainer(
        provider_name=EmbedProvider.COHERE,
        embed=CohereEmbedding(
            api_key=os.getenv("COHERE_API_KEY"),
            model_name=model[EM.NAME],
        ),
        dim=model[EM.DIM],
        alias=model[EM.ALIAS],
    )


def _cohere_text(cfg: EmbedConfig) -> EmbedContainer:
    return _cohere(model=cfg.cohere_embed_model_text, extra="text")


def _cohere_image(cfg: EmbedConfig) -> EmbedContainer:
    return _cohere(model=cfg.cohere_embed_model_image, extra="image")


def _clip(model: dict[str, Any], extra: str, device: str) -> EmbedContainer:
    try:
        from llama_index.embeddings.clip import ClipEmbedding  # type: ignore
    except ImportError:
        raise ImportError(
            EXTRA_PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-embeddings-clip",
                extra=extra,
                feature="ClipEmbedding",
            )
        )

    from .embed_manager import EmbedContainer

    logger.debug(f"Initializing CLIP Embedding with model: {model[EM.NAME]}")

    return EmbedContainer(
        provider_name=EmbedProvider.CLIP,
        embed=ClipEmbedding(
            model_name=model[EM.NAME],
            device=device,
        ),
        dim=model[EM.DIM],
        alias=model[EM.ALIAS],
    )


def _clip_text(cfg: ConfigManager) -> EmbedContainer:
    return _clip(
        model=cfg.embed.clip_embed_model_text,
        extra="localmodel",
        device=cfg.general.device,
    )


def _clip_image(cfg: ConfigManager) -> EmbedContainer:
    return _clip(
        model=cfg.embed.clip_embed_model_image,
        extra="localmodel",
        device=cfg.general.device,
    )


def _clap(model: dict[str, Any], device: str) -> EmbedContainer:
    from ..llama_like.embeddings.clap import ClapEmbedding
    from .embed_manager import EmbedContainer

    logger.debug(f"Initializing CLAP Embedding with model: {model[EM.NAME]}")

    return EmbedContainer(
        provider_name=EmbedProvider.CLAP,
        embed=ClapEmbedding(
            model_name=model[EM.NAME],
            device=device,
        ),
        dim=model[EM.DIM],
        alias=model[EM.ALIAS],
    )


def _clap_text(cfg: ConfigManager) -> EmbedContainer:
    return _clap(model=cfg.embed.clap_embed_model_text, device=cfg.general.device)


def _clap_audio(cfg: ConfigManager) -> EmbedContainer:
    return _clap(model=cfg.embed.clap_embed_model_audio, device=cfg.general.device)


def _huggingface(model: dict[str, Any], extra: str, device: str) -> EmbedContainer:
    try:
        from llama_index.embeddings.huggingface import (  # type: ignore
            HuggingFaceEmbedding,
        )
    except ImportError:
        raise ImportError(
            EXTRA_PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-embeddings-huggingface",
                extra=extra,
                feature="HuggingFaceEmbedding",
            )
        )

    from .embed_manager import EmbedContainer

    logger.debug(f"Initializing HuggingFace Embedding with model: {model[EM.NAME]}")

    return EmbedContainer(
        provider_name=EmbedProvider.HUGGINGFACE,
        embed=HuggingFaceEmbedding(
            model_name=model[EM.NAME],
            device=device,
            trust_remote_code=True,
        ),
        dim=model[EM.DIM],
        alias=model[EM.ALIAS],
    )


def _huggingface_text(cfg: ConfigManager) -> EmbedContainer:
    return _huggingface(
        model=cfg.embed.huggingface_embed_model_text,
        extra="localmodel",
        device=cfg.general.device,
    )


def _huggingface_image(cfg: ConfigManager) -> EmbedContainer:
    return _huggingface(
        model=cfg.embed.huggingface_embed_model_image,
        extra="localmodel",
        device=cfg.general.device,
    )


def _voyage(model: dict[str, Any], extra: str) -> EmbedContainer:
    try:
        from llama_index.embeddings.voyageai.base import VoyageEmbedding  # type: ignore
    except ImportError:
        raise ImportError(
            EXTRA_PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-embeddings-voyageai",
                extra=extra,
                feature="VoyageEmbedding",
            )
        )

    from .embed_manager import EmbedContainer

    return EmbedContainer(
        provider_name=EmbedProvider.VOYAGE,
        embed=VoyageEmbedding(
            api_key=os.getenv("VOYAGE_API_KEY"),
            model_name=model[EM.NAME],
            truncation=False,
            output_dimension=model[EM.DIM],
        ),
        dim=model[EM.DIM],
        alias=model[EM.ALIAS],
    )


def _voyage_text(cfg: EmbedConfig) -> EmbedContainer:
    return _voyage(model=cfg.voyage_embed_model_text, extra="text")


def _voyage_image(cfg: EmbedConfig) -> EmbedContainer:
    return _voyage(model=cfg.voyage_embed_model_image, extra="image")


def _bedrock(model: dict[str, Any]) -> EmbedContainer:
    from ..llama_like.embeddings.bedrock import MultiModalBedrockEmbedding
    from .embed_manager import EmbedContainer

    if model[EM.NAME].startswith("amazon.titan-"):
        kwargs = {"dimensions": model[EM.DIM]}
    elif model[EM.NAME].startswith("amazon.nova-"):
        kwargs = {"embedding_dimension": model[EM.DIM]}
    else:
        kwargs = {}

    return EmbedContainer(
        provider_name=EmbedProvider.BEDROCK,
        embed=MultiModalBedrockEmbedding(
            model_name=model[EM.NAME],
            profile_name=os.getenv("AWS_PROFILE"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            region_name=os.getenv("AWS_REGION") or "us-east-1",
            additional_kwargs=kwargs,
        ),
        dim=model[EM.DIM],
        alias=model[EM.ALIAS],
    )


def _bedrock_text(cfg: EmbedConfig) -> EmbedContainer:
    return _bedrock(cfg.bedrock_embed_model_text)


def _bedrock_image(cfg: EmbedConfig) -> EmbedContainer:
    return _bedrock(cfg.bedrock_embed_model_image)


def _bedrock_audio(cfg: EmbedConfig) -> EmbedContainer:
    return _bedrock(cfg.bedrock_embed_model_audio)


def _bedrock_video(cfg: EmbedConfig) -> EmbedContainer:
    return _bedrock(cfg.bedrock_embed_model_video)
