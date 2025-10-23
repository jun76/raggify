from __future__ import annotations

from ..config import cfg
from ..config.default_settings import EmbedProvider
from ..llama.core.schema import Modality
from .embed_manager import EmbedContainer, EmbedManager

__all__ = ["create_embed_manager"]


def create_embed_manager() -> EmbedManager:
    """埋め込み管理のインスタンスを生成する。

    Raises:
        RuntimeError: インスタンス生成に失敗またはプロバイダ指定漏れ

    Returns:
        EmbedManager: 埋め込み管理
    """
    try:
        conts: dict[Modality, EmbedContainer] = {}
        if cfg.general.text_embed_provider:
            match cfg.general.text_embed_provider:
                case EmbedProvider.OPENAI:
                    cont = _openai_text()
                case EmbedProvider.COHERE:
                    cont = _cohere_text()
                case EmbedProvider.CLIP:
                    cont = _clip_text()
                case EmbedProvider.HUGGINGFACE:
                    cont = _huggingface_text()
                case _:
                    raise ValueError(
                        "unsupported text embed provider: "
                        f"{cfg.general.text_embed_provider}"
                    )
            conts[Modality.TEXT] = cont

        if cfg.general.image_embed_provider:
            match cfg.general.image_embed_provider:
                case EmbedProvider.COHERE:
                    cont = _cohere_image()
                case EmbedProvider.CLIP:
                    cont = _clip_image()
                case EmbedProvider.HUGGINGFACE:
                    cont = _huggingface_image()
                case _:
                    raise ValueError(
                        "unsupported image embed provider: "
                        f"{cfg.general.image_embed_provider}"
                    )
            conts[Modality.IMAGE] = cont

        if cfg.general.audio_embed_provider:
            match cfg.general.audio_embed_provider:
                case EmbedProvider.CLAP:
                    cont = _clap_audio()
                case _:
                    raise ValueError(
                        "unsupported audio embed provider: "
                        f"{cfg.general.audio_embed_provider}"
                    )
            conts[Modality.AUDIO] = cont
    except Exception as e:
        raise RuntimeError(f"failed to create embedding: {e}") from e

    if not conts:
        raise RuntimeError("no embedding providers are specified")

    return EmbedManager(conts)


# 以下、プロバイダ毎のコンテナ生成ヘルパー
def _openai_text() -> EmbedContainer:
    from llama_index.embeddings.openai.base import OpenAIEmbedding

    return EmbedContainer(
        provider_name=EmbedProvider.OPENAI,
        embed=OpenAIEmbedding(
            model=cfg.embed.openai_embed_model_text,
            api_base=cfg.general.openai_base_url,
            # device=cfg.general.device,
        ),
    )


def _cohere_text() -> EmbedContainer:
    from llama_index.embeddings.cohere.base import CohereEmbedding

    return EmbedContainer(
        provider_name=EmbedProvider.COHERE,
        embed=CohereEmbedding(
            model_name=cfg.embed.cohere_embed_model_text,
            # device=cfg.general.device,
        ),
    )


def _cohere_image() -> EmbedContainer:
    from llama_index.embeddings.cohere.base import CohereEmbedding

    return EmbedContainer(
        provider_name=EmbedProvider.COHERE,
        embed=CohereEmbedding(
            model_name=cfg.embed.cohere_embed_model_image,
            device=cfg.general.device,
        ),
    )


def _clip_text() -> EmbedContainer:
    from llama_index.embeddings.clip import ClipEmbedding

    return EmbedContainer(
        provider_name=EmbedProvider.CLIP,
        embed=ClipEmbedding(
            model_name=cfg.embed.clip_embed_model_text,
            device=cfg.general.device,
        ),
    )


def _clip_image() -> EmbedContainer:
    from llama_index.embeddings.clip import ClipEmbedding

    return EmbedContainer(
        provider_name=EmbedProvider.CLIP,
        embed=ClipEmbedding(
            model_name=cfg.embed.clip_embed_model_image,
            device=cfg.general.device,
        ),
    )


def _huggingface_text() -> EmbedContainer:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return EmbedContainer(
        provider_name=EmbedProvider.HUGGINGFACE,
        embed=HuggingFaceEmbedding(
            model_name=cfg.embed.huggingface_embed_model_text,
            device=cfg.general.device,
        ),
    )


def _huggingface_image() -> EmbedContainer:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    return EmbedContainer(
        provider_name=EmbedProvider.HUGGINGFACE,
        embed=HuggingFaceEmbedding(
            model_name=cfg.embed.huggingface_embed_model_image,
            device=cfg.general.device,
            trust_remote_code=True,
        ),
    )


def _clap_audio() -> EmbedContainer:
    from ..llama.embeddings.clap import ClapEmbedding

    return EmbedContainer(
        provider_name=EmbedProvider.CLAP,
        embed=ClapEmbedding(
            model_name=cfg.embed.clap_embed_model_audio,
            device=cfg.general.device,
        ),
    )
