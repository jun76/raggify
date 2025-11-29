from __future__ import annotations

import pytest

from raggify.ingest.loader.file_reader.audio_reader import AudioReader
from raggify.ingest.loader.file_reader.dummy_media_reader import DummyMediaReader
from raggify.ingest.loader.file_reader.pdf_reader import MultiPDFReader
from raggify.ingest.loader.file_reader.video_reader import VideoReader


def test_audio_reader_converts_audio():
    reader = AudioReader()
    docs = list(reader.lazy_load_data("tests/data/audios/sample.wav"))

    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["file_path"].endswith(".mp3")
    assert meta["base_source"].endswith("sample.wav")


def test_audio_reader_skips_invalid_ext():
    reader = AudioReader()
    docs = list(reader.lazy_load_data("tests/data/texts/sample.c"))
    assert docs == []


def test_video_reader_splits_frames_and_audio():
    reader = VideoReader()
    docs = list(reader.lazy_load_data("tests/data/videos/sample.mp4"))

    assert len(docs) >= 1
    file_paths = [doc.metadata["file_path"] for doc in docs]
    assert any(path.endswith(".png") for path in file_paths)
    assert any(path.endswith(".wav") for path in file_paths)


def test_video_reader_invalid_extension():
    reader = VideoReader()
    with pytest.raises(ValueError):
        list(reader.lazy_load_data("tests/data/texts/sample.c"))


def test_pdf_reader_loads_documents():
    reader = MultiPDFReader()
    docs = list(reader.lazy_load_data("tests/data/texts/sample.pdf"))
    assert len(docs) > 0
    text_docs = [doc for doc in docs if doc.text]
    assert text_docs


def test_dummy_media_reader_pass_through():
    reader = DummyMediaReader()
    docs = list(reader.lazy_load_data("tests/data/videos/sample.mp4"))
    assert len(docs) == 1
    assert docs[0].metadata["file_path"].endswith("sample.mp4")


def test_dummy_media_reader_invalid_ext():
    reader = DummyMediaReader()
    docs = list(reader.lazy_load_data("tests/data/texts/sample.c"))
    assert docs == []
