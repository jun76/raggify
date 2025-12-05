from __future__ import annotations

import asyncio

from llama_index.core.schema import Document

from raggify.config.ingest_config import IngestConfig
from raggify.ingest.loader.html_reader.default_html_reader import DefaultHTMLReader
from raggify.ingest.loader.html_reader.html_reader import HTMLReader
from typing import Type, cast

from tests.utils.mock_parser import DummyParser
from tests.utils.mock_reader import patch_html_asset_download, patch_html_temp_file

from .config import configure_test_env

configure_test_env()


class _DummyHTMLReader(HTMLReader):
    async def aload_data(self, url: str) -> list[Document]:
        return []


def _make_reader(cls: Type[HTMLReader] = _DummyHTMLReader, **cfg_overrides) -> HTMLReader:
    ingest_cfg = IngestConfig()
    for key, value in cfg_overrides.items():
        setattr(ingest_cfg, key, value)

    return cls(
        cfg=ingest_cfg,
        asset_url_cache=set(),
        parser=DummyParser(),
    )


def test_cleanse_html_content_strips_and_includes():
    reader = _make_reader(
        include_selectors=["article"],
        exclude_selectors=["nav", ".ads"],
        strip_tags=["script", "style"],
    )
    html = """
<html>
    <body>
        <nav>nav</nav>
        <article><p>Keep me</p><script>alert('x');</script></article>
        <div class='ads'>ads</div>
    </body>
</html>
"""
    cleansed = reader._cleanse_html_text(html)
    assert "Keep me" in cleansed
    assert "nav" not in cleansed
    assert "ads" not in cleansed
    assert "script" not in cleansed


def test_gather_asset_links_filters_and_normalizes():
    reader = cast(DefaultHTMLReader, _make_reader(cls=DefaultHTMLReader))
    html = """
<html>
    <body>
        <img src='/images/img01.png'>
        <img src='https://other.com/img02.png'>
        <a href='/files/audio.mp3'>audio</a>
        <source srcset='/images/img03.png 1x, /images/img04.png 2x'>
    </body>
</html>
"""
    links = reader._gather_asset_links(
        html=html,
        base_url="https://example.com/blog/post",
        allowed_exts={".png", ".mp3"},
        limit=10,
    )
    assert "https://example.com/images/img01.png" in links
    assert "https://example.com/files/audio.mp3" in links
    assert any(link.endswith("img03.png") for link in links)
    assert all(link.startswith("https://example.com") for link in links)


def test_register_asset_url_avoids_duplicates():
    reader = _make_reader()
    assert reader.register_asset_url("https://example.com/a.png") is True
    assert reader.register_asset_url("https://example.com/a.png") is False


def test_adownload_direct_linked_file_success(tmp_path, monkeypatch):
    reader = _make_reader()

    temp_path = tmp_path / "asset.bin"
    patch_html_asset_download(monkeypatch, b"payload")
    patch_html_temp_file(monkeypatch, temp_path)

    result = asyncio.run(
        reader._adownload_direct_linked_file(
            url="https://example.com/img.png",
            allowed_exts={".png"},
            max_asset_bytes=1024,
        )
    )

    assert result == str(temp_path)
    assert temp_path.read_bytes() == b"payload"


def test_adownload_direct_linked_file_rejects_invalid(tmp_path, monkeypatch):
    reader = _make_reader()

    # unsupported extension
    res = asyncio.run(
        reader._adownload_direct_linked_file(
            url="https://example.com/file.exe",
            allowed_exts={".png"},
            max_asset_bytes=1024,
        )
    )
    assert res is None


def test_default_html_reader_aload_assets_keeps_all(monkeypatch):
    reader = cast(DefaultHTMLReader, _make_reader(cls=DefaultHTMLReader))

    def fake_gather(self, html, base_url, allowed_exts, limit=20):
        return [
            "https://example.com/assets/img01.png",
            "https://example.com/assets/img02.png",
        ]

    async def fake_aload(self, url, base_url=None, max_asset_bytes=0):
        return [Document(text=url, metadata={})]

    monkeypatch.setattr(DefaultHTMLReader, "_gather_asset_links", fake_gather)
    monkeypatch.setattr(
        DefaultHTMLReader,
        "aload_direct_linked_file",
        fake_aload,
    )

    docs = asyncio.run(
        reader._aload_assets(
            url="https://example.com/post",
            html="<html></html>",
        )
    )

    texts = [doc.text for doc in docs]
    assert texts == [
        "https://example.com/assets/img01.png",
        "https://example.com/assets/img02.png",
    ]

    patch_html_asset_download(monkeypatch, b"<html></html>", content_type="text/html")

    res = asyncio.run(
        reader._adownload_direct_linked_file(
            url="https://example.com/file.png",
            allowed_exts={".png"},
            max_asset_bytes=1024,
        )
    )
    assert res is None
