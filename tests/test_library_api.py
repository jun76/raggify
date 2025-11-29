from __future__ import annotations

import asyncio
from pathlib import Path

from tests.utils import mock_library_api as mock

SAMPLE_TEXT = Path("tests/data/texts/sample.c").resolve()
SAMPLE_IMAGE = SAMPLE_TEXT
SAMPLE_AUDIO = SAMPLE_TEXT
SAMPLE_VIDEO = SAMPLE_TEXT
DUMMY_SITE = "https://some.site.com"


def test_ingest_sync_calls() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        mock.ingest_path(str(SAMPLE_TEXT))
        mock.ingest_path_list([str(SAMPLE_TEXT)])
        mock.ingest_url(DUMMY_SITE)
        mock.ingest_url_list([DUMMY_SITE])
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_ingest_async_calls() -> None:
    asyncio.run(mock.aingest_path(str(SAMPLE_TEXT)))
    asyncio.run(mock.aingest_path_list([str(SAMPLE_TEXT)]))
    asyncio.run(mock.aingest_url(DUMMY_SITE))
    asyncio.run(mock.aingest_url_list([DUMMY_SITE]))


def test_query_sync_calls() -> None:
    mock.query_text_text("hello")
    mock.query_text_image("hello")
    mock.query_image_image(str(SAMPLE_IMAGE))
    mock.query_text_audio("hello")
    mock.query_audio_audio(str(SAMPLE_AUDIO))
    mock.query_text_video("hello")
    mock.query_image_video(str(SAMPLE_IMAGE))
    mock.query_audio_video(str(SAMPLE_AUDIO))
    mock.query_video_video(str(SAMPLE_VIDEO))


def test_query_async_calls() -> None:
    asyncio.run(mock.aquery_text_text("hello"))
    asyncio.run(mock.aquery_text_image("hello"))
    asyncio.run(mock.aquery_image_image(str(SAMPLE_IMAGE)))
    asyncio.run(mock.aquery_text_audio("hello"))
    asyncio.run(mock.aquery_audio_audio(str(SAMPLE_AUDIO)))
    asyncio.run(mock.aquery_text_video("hello"))
    asyncio.run(mock.aquery_image_video(str(SAMPLE_IMAGE)))
    asyncio.run(mock.aquery_audio_video(str(SAMPLE_AUDIO)))
    asyncio.run(mock.aquery_video_video(str(SAMPLE_VIDEO)))
