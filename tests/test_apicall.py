from __future__ import annotations

import asyncio
from pathlib import Path

from tests.utils import mock_runtime as mr

SAMPLE_TEXT = Path("tests/data/texts/sample.c").resolve()
SAMPLE_IMAGE = SAMPLE_TEXT
SAMPLE_AUDIO = SAMPLE_TEXT
SAMPLE_VIDEO = SAMPLE_TEXT
DUMMY_SITE = "https://some.site.com"


def test_ingest_sync_calls() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        mr.ingest_path(str(SAMPLE_TEXT))
        mr.ingest_path_list([str(SAMPLE_TEXT)])
        mr.ingest_url(DUMMY_SITE)
        mr.ingest_url_list([DUMMY_SITE])
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_ingest_async_calls() -> None:
    asyncio.run(mr.aingest_path(str(SAMPLE_TEXT)))
    asyncio.run(mr.aingest_path_list([str(SAMPLE_TEXT)]))
    asyncio.run(mr.aingest_url(DUMMY_SITE))
    asyncio.run(mr.aingest_url_list([DUMMY_SITE]))


def test_query_sync_calls() -> None:
    mr.query_text_text("hello")
    mr.query_text_image("hello")
    mr.query_image_image(str(SAMPLE_IMAGE))
    mr.query_text_audio("hello")
    mr.query_audio_audio(str(SAMPLE_AUDIO))
    mr.query_text_video("hello")
    mr.query_image_video(str(SAMPLE_IMAGE))
    mr.query_audio_video(str(SAMPLE_AUDIO))
    mr.query_video_video(str(SAMPLE_VIDEO))


def test_query_async_calls() -> None:
    asyncio.run(mr.aquery_text_text("hello"))
    asyncio.run(mr.aquery_text_image("hello"))
    asyncio.run(mr.aquery_image_image(str(SAMPLE_IMAGE)))
    asyncio.run(mr.aquery_text_audio("hello"))
    asyncio.run(mr.aquery_audio_audio(str(SAMPLE_AUDIO)))
    asyncio.run(mr.aquery_text_video("hello"))
    asyncio.run(mr.aquery_image_video(str(SAMPLE_IMAGE)))
    asyncio.run(mr.aquery_audio_video(str(SAMPLE_AUDIO)))
    asyncio.run(mr.aquery_video_video(str(SAMPLE_VIDEO)))
