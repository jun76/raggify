from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..config.config_manager import ConfigManager
from ..config.llm_config import LLMProvider
from ..logger import logger

if TYPE_CHECKING:
    from .llm_manager import LLMContainer, LLMManager

__all__ = ["create_llm_manager"]


def create_llm_manager(cfg: ConfigManager) -> LLMManager:
    """Create an LLM manager instance.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        RuntimeError: If instantiation fails.

    Returns:
        LLMManager: LLM manager.
    """
    from .llm_manager import LLMManager, LLMUsage

    try:
        conts: dict[LLMUsage, LLMContainer] = {}
        if cfg.general.image_caption_transform_provider:
            conts[LLMUsage.IMAGE_CAPTIONER] = _create_image_captioner(cfg)
        if cfg.general.audio_caption_transform_provider:
            conts[LLMUsage.AUDIO_CAPTIONER] = _create_audio_captioner(cfg)
        if cfg.general.video_caption_transform_provider:
            conts[LLMUsage.VIDEO_CAPTIONER] = _create_video_captioner(cfg)
    except (ValueError, ImportError) as e:
        raise RuntimeError("invalid LLM settings") from e
    except Exception as e:
        raise RuntimeError("failed to create LLMs") from e

    if not conts:
        logger.info("no LLM providers are specified")

    return LLMManager(conts)


def _create_image_captioner(cfg: ConfigManager) -> LLMContainer:
    """Create image caption transform container.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        ValueError: If image caption transform provider is not specified or unsupported.

    Returns:
        LLMContainer: Image caption transform container.
    """
    provider = cfg.general.image_caption_transform_provider
    if provider is None:
        raise ValueError("image caption transform provider is not specified")
    match provider:
        case LLMProvider.OPENAI:
            return _openai_image_captioner(cfg)
        case _:
            raise ValueError(
                f"unsupported image caption transform provider: {provider}"
            )


def _create_audio_captioner(cfg: ConfigManager) -> LLMContainer:
    """Create audio caption transform container.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        ValueError: If audio caption transform provider is not specified or unsupported.

    Returns:
        LLMContainer: Audio caption transform container.
    """
    provider = cfg.general.audio_caption_transform_provider
    if provider is None:
        raise ValueError("audio caption transform provider is not specified")
    match provider:
        case LLMProvider.OPENAI:
            return _openai_audio_captioner(cfg)
        case _:
            raise ValueError(
                f"unsupported audio caption transform provider: {provider}"
            )


def _create_video_captioner(cfg: ConfigManager) -> LLMContainer:
    """Create video caption transform container.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        ValueError: If video caption transform provider is not specified or unsupported.

    Returns:
        LLMContainer: Video caption transform container.
    """
    provider = cfg.general.video_caption_transform_provider
    if provider is None:
        raise ValueError("video caption transform provider is not specified")
    match provider:
        case LLMProvider.OPENAI:
            return _openai_video_captioner(cfg)
        case _:
            raise ValueError(
                f"unsupported video caption transform provider: {provider}"
            )


# Container generation helpers per provider
def _openai(
    model: str, api_base: Optional[str], modalities: Optional[list[str]] = None
) -> LLMContainer:
    from llama_index.llms.openai import OpenAI

    from .llm_manager import LLMContainer

    return LLMContainer(
        provider_name=LLMProvider.OPENAI,
        llm=OpenAI(
            model=model,
            api_base=api_base,
            temperature=0,
            modalities=modalities,
        ),
    )


def _openai_image_captioner(cfg: ConfigManager) -> LLMContainer:
    return _openai(
        model=cfg.llm.openai_image_caption_transform_model,
        api_base=cfg.general.openai_base_url,
    )


def _openai_audio_captioner(cfg: ConfigManager) -> LLMContainer:
    return _openai(
        model=cfg.llm.openai_audio_caption_transform_model,
        api_base=cfg.general.openai_base_url,
        modalities=["text"],
    )


def _openai_video_captioner(cfg: ConfigManager) -> LLMContainer:
    return _openai(
        model=cfg.llm.openai_video_caption_transform_model,
        api_base=cfg.general.openai_base_url,
        modalities=["text"],
    )
