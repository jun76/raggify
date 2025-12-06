from __future__ import annotations

from typing import TYPE_CHECKING

from ..config.config_manager import ConfigManager
from ..config.llm_config import LLMProvider
from ..core.const import PKG_NOT_FOUND_MSG
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
            conts[LLMUsage.TEXT_SUMMARIZER] = _create_text_summarize_transform(cfg)
        if cfg.general.image_summarize_transform_provider:
            conts[LLMUsage.IMAGE_SUMMARIZER] = _create_image_summarize_transform(cfg)
    except (ValueError, ImportError) as e:
        raise RuntimeError("invalid LLM settings") from e
    except Exception as e:
        raise RuntimeError("failed to create LLMs") from e

    if not conts:
        logger.info("no LLM providers are specified")

    return LLMManager(conts)


def _create_text_summarize_transform(cfg: ConfigManager) -> LLMContainer:
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
            return _openai_text_summarize_transform(cfg)
        case LLMProvider.HUGGINGFACE:
            return _huggingface_text_summarize_transform(cfg)
        case _:
            raise ValueError(
                f"unsupported text summarize transform provider: {provider}"
            )


def _create_image_summarize_transform(cfg: ConfigManager) -> LLMContainer:
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
            return _openai_image_summarize_transform(cfg)
        case LLMProvider.HUGGINGFACE:
            return _huggingface_image_summarize_transform(cfg)
        case _:
            raise ValueError(
                f"unsupported image summarize transform provider: {provider}"
            )


# Container generation helpers per provider
def _openai_text_summarize_transform(cfg: ConfigManager) -> LLMContainer:
    from llama_index.multi_modal_llms.openai import OpenAIMultiModal

    from .llm_manager import LLMContainer

    return LLMContainer(
        provider_name=LLMProvider.OPENAI,
        llm=OpenAIMultiModal(
            model=cfg.llm.openai_text_summarize_transform_model,
            api_base=cfg.general.openai_base_url,
            temperature=0,
        ),
    )


def _huggingface_text_summarize_transform(cfg: ConfigManager) -> LLMContainer:
    try:
        # FIXME: issue #6 HuggingFaceMultiModal version mismatch
        from llama_index.multi_modal_llms.huggingface import (
            HuggingFaceMultiModal,  # type: ignore
        )
    except ImportError as e:
        raise ImportError(
            PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-multi-modal-llms-huggingface",
                extra="localmodel",
                feature="HuggingFaceMultiModal",
            )
        ) from e

    from .llm_manager import LLMContainer

    return LLMContainer(
        provider_name=LLMProvider.HUGGINGFACE,
        llm=HuggingFaceMultiModal.from_model_name(
            model_name=cfg.llm.huggingface_text_summarize_transform_model,
            device=cfg.general.device,
            temperature=0,
        ),
    )


def _openai_image_summarize_transform(cfg: ConfigManager) -> LLMContainer:
    from llama_index.multi_modal_llms.openai import OpenAIMultiModal

    from .llm_manager import LLMContainer

    return LLMContainer(
        provider_name=LLMProvider.OPENAI,
        llm=OpenAIMultiModal(
            model=cfg.llm.openai_image_summarize_transform_model,
            api_base=cfg.general.openai_base_url,
            temperature=0,
        ),
    )


def _huggingface_image_summarize_transform(cfg: ConfigManager) -> LLMContainer:
    try:
        from llama_index.multi_modal_llms.huggingface import (
            HuggingFaceMultiModal,  # type: ignore
        )
    except ImportError as e:
        raise ImportError(
            PKG_NOT_FOUND_MSG.format(
                pkg="llama-index-multi-modal-llms-huggingface",
                extra="localmodel",
                feature="HuggingFaceMultiModal",
            )
        ) from e

    from .llm_manager import LLMContainer

    return LLMContainer(
        provider_name=LLMProvider.HUGGINGFACE,
        llm=HuggingFaceMultiModal.from_model_name(
            model_name=cfg.llm.huggingface_image_summarize_transform_model,
            device=cfg.general.device,
            temperature=0,
        ),
    )
