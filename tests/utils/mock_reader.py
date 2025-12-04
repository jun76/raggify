from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, patch


def patch_audio_convert(monkeypatch, output_path: Path) -> None:
    """Patch AudioReader._convert to avoid ffmpeg calls."""
    from raggify.ingest.loader.file_reader.audio_reader import AudioReader

    def _fake_convert(self, src: str) -> Path:
        return Path(output_path)

    monkeypatch.setattr(AudioReader, "_convert", _fake_convert)


def patch_video_extract(
    monkeypatch,
    frame_paths: Iterable[Path],
    audio_path: Path | None,
) -> None:
    """Patch frame/audio extraction for VideoReader."""
    from raggify.ingest.loader.file_reader.video_reader import VideoReader

    frames = [Path(p) for p in frame_paths]

    def _fake_frames(self, src: str) -> list[Path]:
        return frames

    def _fake_audio(self, src: str) -> Path | None:
        return Path(audio_path) if audio_path else None

    monkeypatch.setattr(VideoReader, "_extract_frames", _fake_frames)
    monkeypatch.setattr(VideoReader, "_extract_audio", _fake_audio)


def patch_html_temp_file(monkeypatch, path: Path) -> None:
    """Patch get_temp_file_path_from used by HTMLReader."""
    from raggify.core import utils as core_utils

    def _fake_temp(source: str, suffix: str) -> str:
        return str(path)

    monkeypatch.setattr(core_utils, "get_temp_file_path_from", _fake_temp)


def patch_html_asset_download(monkeypatch, content: bytes, content_type: str = "image/png") -> None:
    """Patch arequest_get for HTMLReader asset downloads."""

    payload = content

    async def _fake_get(*args, **kwargs):
        class _Resp:
            headers = {"Content-Type": content_type}
            content = payload

        return _Resp()

    monkeypatch.setattr(
        "raggify.ingest.loader.html_reader.html_reader.arequest_get",
        AsyncMock(side_effect=_fake_get),
    )


@contextmanager
def patch_html_fetchers() -> Iterator[None]:
    """Patch HTML fetchers (sitemap/default/Wikipedia HTML) and asset download."""
    with ExitStack() as stack:
        from llama_index.core.schema import Document

        async def fake_afetch_text(url: str, *args, **kwargs) -> str:
            if url.endswith(".xml"):
                return """<?xml version="1.0" encoding="UTF-8"?>
<?xml-stylesheet type="text/xsl" href="https://some.site.com/wp-sitemap.xsl" ?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://some.site.com/blog/aaa/</loc></url>
  <url><loc>https://some.site.com/blog/bbb/</loc></url>
  <url><loc>https://some.site.com/blog/ccc/</loc></url>
</urlset>
"""
            if "wikipedia.org" in url:
                return "<html><body><p>Sample Wikipedia content.</p></body></html>"

            return "<html>Sample content.</html>"

        stack.enter_context(
            patch(
                "raggify.ingest.loader.util.afetch_text",
                new=AsyncMock(side_effect=fake_afetch_text),
            )
        )

        async def fake_adownload_direct_linked_file(
            url: str,
            allowed_exts: set[str],
            max_asset_bytes: int,
        ) -> Optional[str]:
            tmp_path = Path("/tmp/tmp_raggify_mock_asset.png")
            try:
                import base64

                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                png_bytes = base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
                )
                tmp_path.write_bytes(png_bytes)
            except OSError:
                return None
            return str(tmp_path)

        stack.enter_context(
            patch(
                "raggify.ingest.loader.html_reader.html_reader.HTMLReader._adownload_direct_linked_file",
                new=AsyncMock(side_effect=fake_adownload_direct_linked_file),
            )
        )

        async def fake_aload_direct_linked_file(
            url: str,
            base_url: Optional[str] = None,
            max_asset_bytes: int = 100 * 1024 * 1024,
        ) -> Document:
            meta = {
                "url": url,
                "base_source": base_url or "",
                "file_path": "/tmp/tmp_raggify_mock_asset.png",
                "temp_file_path": "/tmp/tmp_raggify_mock_asset.png",
            }
            return Document(text="", metadata=meta)

        stack.enter_context(
            patch(
                "raggify.ingest.loader.html_reader.html_reader.HTMLReader.aload_direct_linked_file",
                new=AsyncMock(side_effect=fake_aload_direct_linked_file),
            )
        )
        yield


@dataclass
class DummyWikipediaPage:
    pageid: str
    content: str
    url: str
    images: list[str]


@contextmanager
def patch_wikipedia_reader(
    *,
    content: str = "Sample Wikipedia content.",
    images: Optional[list[str]] = None,
) -> Iterator[None]:
    """Patch MultiWikipediaReader to return dummy Wikipedia data."""
    from raggify.ingest.loader.html_reader.wikipedia_reader import MultiWikipediaReader

    imgs = images or ["https://some.site.com/sample.png"]

    def _fake_fetch(self, url: str, **kwargs) -> DummyWikipediaPage:
        return DummyWikipediaPage(
            pageid="demo-page",
            content=content,
            url=url,
            images=imgs,
        )

    with patch.object(MultiWikipediaReader, "_fetch_wiki_page", _fake_fetch):
        yield
