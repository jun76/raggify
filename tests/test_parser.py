from __future__ import annotations

from pathlib import Path

from raggify.config.embed_config import EmbedProvider
from raggify.config.general_config import GeneralConfig
from raggify.core.exts import Exts
from raggify.ingest.loader import DefaultParser
from raggify.ingest.loader.file_reader import AudioReader, VideoReader

from .config import configure_test_env

configure_test_env()

SAMPLE_TEXT = Path("tests/data/texts/sample.txt").resolve()


def _make_parser(cfg: GeneralConfig, exts: set[str]) -> DefaultParser:
    return DefaultParser(cfg=cfg, ingest_target_exts=exts)


def test_parse_text_file_returns_document() -> None:
    parser = _make_parser(GeneralConfig(), {".txt"})

    docs = parser.parse(str(SAMPLE_TEXT))

    assert docs
    assert Path(docs[0].metadata["file_path"]).name == "sample.txt"


def test_parse_skips_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.xyz"
    file_path.write_text("dummy", encoding="utf-8")
    parser = _make_parser(GeneralConfig(), {".txt"})

    docs = parser.parse(str(file_path))

    assert docs == []


def test_audio_reader_registered_when_audio_provider_enabled() -> None:
    cfg = GeneralConfig(audio_embed_provider=EmbedProvider.CLAP)
    parser = _make_parser(cfg, {".mp3"})

    for ext in Exts.AUDIO:
        assert isinstance(parser._readers[ext], AudioReader)  # type: ignore[attr-defined]


def test_video_reader_registered_with_fallback() -> None:
    cfg = GeneralConfig(video_embed_provider=None, use_modality_fallback=True)
    parser = _make_parser(cfg, {".mp4"})

    for ext in Exts.VIDEO:
        assert isinstance(parser._readers[ext], VideoReader)  # type: ignore[attr-defined]
