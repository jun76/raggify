from __future__ import annotations

import asyncio

from raggify.config.ingest_config import IngestConfig
from raggify.ingest.loader.html_reader.html_reader import HTMLReader
from tests.utils.mock_reader import patch_html_asset_download, patch_html_temp_file

from .config import configure_test_env

configure_test_env()


def _make_reader(**cfg_overrides):
    cfg = IngestConfig()
    for key, value in cfg_overrides.items():
        setattr(cfg, key, value)
    return HTMLReader(
        cfg=cfg, asset_url_cache=set(), ingest_target_exts={".png", ".mp3"}
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
    cleansed = reader.cleanse_html_content(html)
    assert "Keep me" in cleansed
    assert "nav" not in cleansed
    assert "ads" not in cleansed
    assert "script" not in cleansed


def test_gather_asset_links_filters_and_normalizes():
    reader = _make_reader()
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
    links = reader.gather_asset_links(
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

    patch_html_asset_download(monkeypatch, b"<html></html>", content_type="text/html")

    res = asyncio.run(
        reader._adownload_direct_linked_file(
            url="https://example.com/file.png",
            allowed_exts={".png"},
            max_asset_bytes=1024,
        )
    )
    assert res is None
