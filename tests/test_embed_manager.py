from __future__ import annotations

import asyncio

import pytest

from raggify.llama_like.core.schema import Modality
from tests.utils.mock_embed import (
    DummyAudioBase,
    DummyAudioEmbedding,
    DummyImageEmbedding,
    DummyMultiModalBase,
    DummyTextEmbedding,
    DummyVideoBase,
    DummyVideoEmbedding,
    make_dummy_manager,
)


@pytest.fixture(autouse=True)
def patch_embedding_bases(monkeypatch):
    monkeypatch.setattr(
        "llama_index.core.embeddings.multi_modal_base.MultiModalEmbedding",
        DummyMultiModalBase,
    )
    monkeypatch.setattr(
        "raggify.llama_like.embeddings.multi_modal_base.AudioEmbedding", DummyAudioBase
    )
    monkeypatch.setattr(
        "raggify.llama_like.embeddings.multi_modal_base.VideoEmbedding", DummyVideoBase
    )


def test_name():
    manager = make_dummy_manager({Modality.TEXT: DummyTextEmbedding()})
    assert manager.name == "dummy"


def test_aembed_text_success():
    manager = make_dummy_manager({Modality.TEXT: DummyTextEmbedding()})
    result = asyncio.run(manager.aembed_text(["hello", "world"]))
    assert len(result) == 2
    assert result[0] == [1.0, 1.0]


def test_aembed_text_missing():
    manager = make_dummy_manager({})
    result = asyncio.run(manager.aembed_text(["hi"]))
    assert result == []


def test_aembed_image_success():
    manager = make_dummy_manager({Modality.IMAGE: DummyImageEmbedding()})
    result = asyncio.run(manager.aembed_image(["img.png"]))
    assert len(result) == 1
    assert result[0] == [0.1, 0.1]


def test_aembed_image_wrong_type():
    class NotMultiModal:
        async def aget_image_embedding_batch(
            self, img_file_paths, show_progress: bool = True
        ):
            return []

    manager = make_dummy_manager({Modality.IMAGE: NotMultiModal()})
    with pytest.raises(RuntimeError):
        asyncio.run(manager.aembed_image(["img.png"]))


def test_aembed_audio_success():
    manager = make_dummy_manager({Modality.AUDIO: DummyAudioEmbedding()})
    result = asyncio.run(manager.aembed_audio(["sample.wav"]))
    assert result[0] == [0.2, 0.2]


def test_aembed_audio_wrong_type():
    manager = make_dummy_manager({Modality.AUDIO: DummyImageEmbedding()})
    with pytest.raises(RuntimeError):
        asyncio.run(manager.aembed_audio(["sample.wav"]))


def test_aembed_video_success():
    manager = make_dummy_manager({Modality.VIDEO: DummyVideoEmbedding()})
    result = asyncio.run(manager.aembed_video(["movie.mp4"]))
    assert result[0] == [0.3, 0.3]


def test_aembed_video_wrong_type():
    manager = make_dummy_manager({Modality.VIDEO: DummyAudioEmbedding()})
    with pytest.raises(RuntimeError):
        asyncio.run(manager.aembed_video(["movie.mp4"]))
