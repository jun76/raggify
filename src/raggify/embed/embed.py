from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ..config.config_manager import ConfigManager
from ..config.embed_config import EmbedConfig
from ..config.embed_config import EmbedModel as EM
from ..config.embed_config import EmbedProvider
from ..llama.core.schema import Modality

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
                    cont = _cohere_text(cfg.embed)
                case EmbedProvider.CLIP:
                    cont = _clip_text(cfg)
                case EmbedProvider.HUGGINGFACE:
                    cont = _huggingface_text(cfg)
                case EmbedProvider.VOYAGE:
                    cont = _voyage_text(cfg.embed)
                case EmbedProvider.BEDROCK:
                    cont = _bedrock_text(cfg.embed)
                case _:
                    raise ValueError(
                        "unsupported text embed provider: "
                        f"{cfg.general.text_embed_provider}"
                    )
            conts[Modality.TEXT] = cont

        if cfg.general.image_embed_provider:
            match cfg.general.image_embed_provider:
                case EmbedProvider.COHERE:
                    cont = _cohere_image(cfg.embed)
                case EmbedProvider.CLIP:
                    cont = _clip_image(cfg)
                case EmbedProvider.HUGGINGFACE:
                    cont = _huggingface_image(cfg)
                case EmbedProvider.VOYAGE:
                    cont = _voyage_image(cfg.embed)
                case EmbedProvider.BEDROCK:
                    cont = _bedrock_image(cfg.embed)
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
                case EmbedProvider.BEDROCK:
                    cont = _bedrock_audio(cfg.embed)
                case _:
                    raise ValueError(
                        "unsupported audio embed provider: "
                        f"{cfg.general.audio_embed_provider}"
                    )
            conts[Modality.AUDIO] = cont

        if cfg.general.video_embed_provider:
            match cfg.general.video_embed_provider:
                case EmbedProvider.BEDROCK:
                    cont = _bedrock_video(cfg.embed)
                case _:
                    raise ValueError(
                        "unsupported video embed provider: "
                        f"{cfg.general.video_embed_provider}"
                    )
            conts[Modality.VIDEO] = cont
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


def _cohere(model: dict[str, Any]) -> EmbedContainer:
    from llama_index.embeddings.cohere.base import CohereEmbedding

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
    return _cohere(cfg.cohere_embed_model_text)


def _cohere_image(cfg: EmbedConfig) -> EmbedContainer:
    return _cohere(cfg.cohere_embed_model_image)


def _clip(model: dict[str, Any], device: str) -> EmbedContainer:
    from llama_index.embeddings.clip import ClipEmbedding

    from .embed_manager import EmbedContainer

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
    return _clip(model=cfg.embed.clip_embed_model_text, device=cfg.general.device)


def _clip_image(cfg: ConfigManager) -> EmbedContainer:
    return _clip(model=cfg.embed.clip_embed_model_image, device=cfg.general.device)


def _huggingface(model: dict[str, Any], device: str) -> EmbedContainer:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    from .embed_manager import EmbedContainer

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
        model=cfg.embed.huggingface_embed_model_text, device=cfg.general.device
    )


def _huggingface_image(cfg: ConfigManager) -> EmbedContainer:
    return _huggingface(
        model=cfg.embed.huggingface_embed_model_image, device=cfg.general.device
    )


def _clap_audio(cfg: ConfigManager) -> EmbedContainer:
    from ..llama.embeddings.clap import ClapEmbedding
    from .embed_manager import EmbedContainer

    model = cfg.embed.clap_embed_model_audio

    return EmbedContainer(
        provider_name=EmbedProvider.CLAP,
        embed=ClapEmbedding(
            model_name=model[EM.NAME],
            device=cfg.general.device,
        ),
        dim=model[EM.DIM],
        alias=model[EM.ALIAS],
    )


def _voyage(model: dict[str, Any]) -> EmbedContainer:
    from llama_index.embeddings.voyageai.base import VoyageEmbedding

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
    return _voyage(cfg.voyage_embed_model_text)


def _voyage_image(cfg: EmbedConfig) -> EmbedContainer:
    return _voyage(cfg.voyage_embed_model_image)


def _bedrock(cfg: EmbedConfig, model: dict[str, Any]) -> EmbedContainer:
    from ..llama.embeddings.bedrock import MultiModalBedrockEmbedding
    from .embed_manager import EmbedContainer

    return EmbedContainer(
        provider_name=EmbedProvider.BEDROCK,
        embed=MultiModalBedrockEmbedding(
            model_name=model[EM.NAME],
            profile_name=cfg.bedrock_profile_name,
            aws_access_key_id=cfg.bedrock_aws_access_key_id,
            aws_secret_access_key=cfg.bedrock_aws_secret_access_key,
            aws_session_token=cfg.bedrock_aws_session_token,
            region_name=cfg.bedrock_region_name,
        ),
        dim=model[EM.DIM],
        alias=model[EM.ALIAS],
    )


def _bedrock_text(cfg: EmbedConfig) -> EmbedContainer:
    return _bedrock(cfg, cfg.bedrock_embed_model_text)


def _bedrock_image(cfg: EmbedConfig) -> EmbedContainer:
    return _bedrock(cfg, cfg.bedrock_embed_model_image)


def _bedrock_audio(cfg: EmbedConfig) -> EmbedContainer:
    return _bedrock(cfg, cfg.bedrock_embed_model_audio)


def _bedrock_video(cfg: EmbedConfig) -> EmbedContainer:
    return _bedrock(cfg, cfg.bedrock_embed_model_video)
