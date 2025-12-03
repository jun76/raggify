from __future__ import annotations

import pytest

from raggify.ingest.loader.file_reader.audio_reader import AudioReader
from raggify.ingest.loader.file_reader.dummy_media_reader import DummyMediaReader
from raggify.ingest.loader.file_reader.pdf_reader import MultiPDFReader
from raggify.ingest.loader.file_reader.video_reader import VideoReader
from tests.utils.mock_reader import patch_audio_convert, patch_video_extract

from .config import configure_test_env

configure_test_env()


def test_audio_reader_converts_audio(monkeypatch, tmp_path):
    output = tmp_path / "converted.mp3"
    output.write_bytes(b"audio")
    patch_audio_convert(monkeypatch, output)

    reader = AudioReader()
    docs = list(reader.lazy_load_data("tests/data/audios/sample.wav"))

    assert len(docs) == 1
    meta = docs[0].metadata
    assert meta["file_path"] == str(output)
    assert meta["base_source"].endswith("sample.wav")


def test_audio_reader_skips_invalid_ext(monkeypatch, tmp_path):
    patch_audio_convert(monkeypatch, tmp_path / "out.mp3")
    reader = AudioReader()
    docs = list(reader.lazy_load_data("tests/data/texts/sample.c"))
    assert docs == []


def test_video_reader_splits_frames_and_audio(monkeypatch, tmp_path):
    frame1 = tmp_path / "frame_00000.png"
    frame2 = tmp_path / "frame_00001.png"
    for p in (frame1, frame2):
        p.write_bytes(b"img")
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"aud")
    patch_video_extract(monkeypatch, [frame1, frame2], audio)

    reader = VideoReader()
    docs = list(reader.lazy_load_data("tests/data/videos/sample.mp4"))

    assert len(docs) == 3
    image_paths = [doc.metadata["file_path"] for doc in docs[:-1]]
    assert image_paths == [str(frame1), str(frame2)]
    assert docs[-1].metadata["file_path"] == str(audio)


def test_video_reader_invalid_extension(monkeypatch):
    patch_video_extract(monkeypatch, [], None)
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
