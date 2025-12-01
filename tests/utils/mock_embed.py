from __future__ import annotations

from typing import Any

from raggify.embed.embed_manager import EmbedContainer, EmbedManager
from raggify.llama_like.core.schema import Modality


class DummyTextEmbedding:
    async def aget_text_embedding_batch(self, texts, show_progress=True):
        return [[1.0] * 2 for _ in texts]


class DummyMultiModalBase:
    async def aget_image_embedding_batch(self, img_file_paths, show_progress=True):
        return [[0.1] * 2 for _ in img_file_paths]


class DummyAudioBase:
    async def aget_audio_embedding_batch(self, audio_file_paths, show_progress=True):
        return [[0.2] * 2 for _ in audio_file_paths]


class DummyVideoBase(DummyAudioBase):
    async def aget_video_embedding_batch(self, video_file_paths, show_progress=True):
        return [[0.3] * 2 for _ in video_file_paths]


class DummyImageEmbedding(DummyMultiModalBase):
    pass


class DummyAudioEmbedding(DummyAudioBase):
    pass


class DummyVideoEmbedding(DummyVideoBase):
    pass


def make_dummy_manager(
    mapping: dict[Modality, Any], batch_size: int = 10, batch_interval_sec: int = 0
) -> EmbedManager:
    args = {
        modality: EmbedContainer(
            provider_name="dummy",
            embed=embed,
            dim=2,
            alias=f"alias-{modality.value}",
        )
        for modality, embed in mapping.items()
    }
    return EmbedManager(args, batch_size=batch_size, batch_interval_sec=batch_interval_sec)
