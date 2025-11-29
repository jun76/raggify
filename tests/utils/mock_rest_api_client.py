from __future__ import annotations

from unittest.mock import MagicMock

from raggify_client import RestAPIClient


class RestClientStub(RestAPIClient):
    """RestAPIClient stub backed by mock _make_request."""

    def __init__(self) -> None:
        super().__init__()
        self._make_request = MagicMock(return_value={"status": "ok"})


client_stub = RestClientStub()


def status_client() -> dict[str, str]:
    return client_stub.status()


def reload_client() -> dict[str, str]:
    return client_stub.reload()


def upload_client() -> dict[str, str]:
    return client_stub.upload([("file.bin", b"hello", None)])


def job_client() -> dict[str, str]:
    return client_stub.job()


def ingest_path_client() -> dict[str, str]:
    return client_stub.ingest_path("/tmp/sample.txt")


def ingest_path_list_client() -> dict[str, str]:
    return client_stub.ingest_path_list("/tmp/list.txt")


def ingest_url_client() -> dict[str, str]:
    return client_stub.ingest_url("https://some.site.com")


def ingest_url_list_client() -> dict[str, str]:
    return client_stub.ingest_url_list("/tmp/urls.txt")


def query_text_text_client() -> dict[str, str]:
    return client_stub.query_text_text("hello")


def query_text_image_client() -> dict[str, str]:
    return client_stub.query_text_image("hello")


def query_image_image_client() -> dict[str, str]:
    return client_stub.query_image_image("/tmp/image.png")


def query_text_audio_client() -> dict[str, str]:
    return client_stub.query_text_audio("hello")


def query_audio_audio_client() -> dict[str, str]:
    return client_stub.query_audio_audio("/tmp/audio.wav")


def query_text_video_client() -> dict[str, str]:
    return client_stub.query_text_video("hello")


def query_image_video_client() -> dict[str, str]:
    return client_stub.query_image_video("/tmp/image.png")


def query_audio_video_client() -> dict[str, str]:
    return client_stub.query_audio_video("/tmp/audio.wav")


def query_video_video_client() -> dict[str, str]:
    return client_stub.query_video_video("/tmp/video.mp4")


CLIENT_CALLS = [
    status_client,
    reload_client,
    upload_client,
    job_client,
    ingest_path_client,
    ingest_path_list_client,
    ingest_url_client,
    ingest_url_list_client,
    query_text_text_client,
    query_text_image_client,
    query_image_image_client,
    query_text_audio_client,
    query_audio_audio_client,
    query_text_video_client,
    query_image_video_client,
    query_audio_video_client,
    query_video_video_client,
]
