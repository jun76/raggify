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
        if cfg.general.text_summarize_transform_provider:
            conts[LLMUsage.TEXT_SUMMARIZER] = _create_text_summarizer(cfg)
        if cfg.general.image_summarize_transform_provider:
            conts[LLMUsage.IMAGE_SUMMARIZER] = _create_image_summarizer(cfg)
        if cfg.general.audio_summarize_transform_provider:
            conts[LLMUsage.AUDIO_SUMMARIZER] = _create_audio_summarizer(cfg)
        if cfg.general.video_summarize_transform_provider:
            conts[LLMUsage.VIDEO_SUMMARIZER] = _create_video_summarizer(cfg)
    except (ValueError, ImportError) as e:
        raise RuntimeError("invalid LLM settings") from e
    except Exception as e:
        raise RuntimeError("failed to create LLMs") from e

    if not conts:
        logger.info("no LLM providers are specified")

    return LLMManager(conts)


def _create_text_summarizer(cfg: ConfigManager) -> LLMContainer:
    """Create text summarize transform container.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        ValueError: If text summarize transform provider is not specified or unsupported.

    Returns:
        LLMContainer: Text summarize transform container.
    """
    provider = cfg.general.text_summarize_transform_provider
    if provider is None:
        raise ValueError("text summarize transform provider is not specified")
    match provider:
        case LLMProvider.OPENAI:
            return _openai_text_summarizer(cfg)
        case _:
            raise ValueError(
                f"unsupported text summarize transform provider: {provider}"
            )


def _create_image_summarizer(cfg: ConfigManager) -> LLMContainer:
    """Create image summarize transform container.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        ValueError: If image summarize transform provider is not specified or unsupported.

    Returns:
        LLMContainer: Image summarize transform container.
    """
    provider = cfg.general.image_summarize_transform_provider
    if provider is None:
        raise ValueError("image summarize transform provider is not specified")
    match provider:
        case LLMProvider.OPENAI:
            return _openai_image_summarizer(cfg)
        case _:
            raise ValueError(
                f"unsupported image summarize transform provider: {provider}"
            )


def _create_audio_summarizer(cfg: ConfigManager) -> LLMContainer:
    """Create audio summarize transform container.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        ValueError: If audio summarize transform provider is not specified or unsupported.

    Returns:
        LLMContainer: Audio summarize transform container.
    """
    provider = cfg.general.audio_summarize_transform_provider
    if provider is None:
        raise ValueError("audio summarize transform provider is not specified")
    match provider:
        case LLMProvider.OPENAI:
            return _openai_audio_summarizer(cfg)
        case _:
            raise ValueError(
                f"unsupported audio summarize transform provider: {provider}"
            )


def _create_video_summarizer(cfg: ConfigManager) -> LLMContainer:
    """Create video summarize transform container.

    Args:
        cfg (ConfigManager): Config manager.

    Raises:
        ValueError: If video summarize transform provider is not specified or unsupported.

    Returns:
        LLMContainer: Video summarize transform container.
    """
    provider = cfg.general.video_summarize_transform_provider
    if provider is None:
        raise ValueError("video summarize transform provider is not specified")
    match provider:
        case LLMProvider.OPENAI:
            return _openai_video_summarizer(cfg)
        case _:
            raise ValueError(
                f"unsupported video summarize transform provider: {provider}"
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


def _openai_text_summarizer(cfg: ConfigManager) -> LLMContainer:
    return _openai(
        model=cfg.llm.openai_text_summarize_transform_model,
        api_base=cfg.general.openai_base_url,
    )


def _openai_image_summarizer(cfg: ConfigManager) -> LLMContainer:
    return _openai(
        model=cfg.llm.openai_image_summarize_transform_model,
        api_base=cfg.general.openai_base_url,
    )


def _openai_audio_summarizer(cfg: ConfigManager) -> LLMContainer:
    return _openai(
        model=cfg.llm.openai_audio_summarize_transform_model,
        api_base=cfg.general.openai_base_url,
        modalities=["text"],
    )


def _openai_video_summarizer(cfg: ConfigManager) -> LLMContainer:
    return _openai(
        model=cfg.llm.openai_video_summarize_transform_model,
        api_base=cfg.general.openai_base_url,
        modalities=["text"],
    )
