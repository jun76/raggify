from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.utils.mock_rest_api_server import patch_rest_api_server


def _load_fastapi_module():
    module = importlib.import_module("raggify.server.fastapi")
    return importlib.reload(module)


@pytest.fixture()
def api_client(tmp_path):
    module = _load_fastapi_module()
    with patch_rest_api_server(module=module, upload_dir=tmp_path) as ctx:
        with TestClient(module.app) as client:
            yield client, ctx


def test_status_and_reload(api_client):
    client, ctx = api_client

    resp = client.get("/v1/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["vector store"] == "vector"

    before = ctx.runtime.build.call_count
    reload_resp = client.get("/v1/reload")
    assert reload_resp.status_code == 200
    assert reload_resp.json()["status"] == "ok"
    assert ctx.runtime.build.call_count == before + 1


def test_upload_endpoint(api_client, tmp_path):
    client, _ = api_client

    files = [("files", ("sample.txt", b"hello", "text/plain"))]
    resp = client.post("/v1/upload", files=files)
    assert resp.status_code == 200

    payload = resp.json()
    assert payload["files"][0]["filename"] == "sample.txt"
    saved = Path(payload["files"][0]["save_path"])
    assert saved.exists()
    assert saved.read_bytes() == b"hello"


def test_ingest_and_job_routes(api_client):
    client, _ = api_client

    ingest_requests = [
        ("/v1/ingest/path", {"path": "/tmp/a.txt"}),
        ("/v1/ingest/path_list", {"path": "/tmp/list.txt"}),
        ("/v1/ingest/url", {"url": "https://some.site.com"}),
        ("/v1/ingest/url_list", {"path": "/tmp/urls.txt"}),
    ]

    job_ids = []
    for endpoint, body in ingest_requests:
        resp = client.post(endpoint, json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        job_ids.append(data["job_id"])

    job_id = job_ids[0]
    detail = client.post("/v1/job", json={"job_id": job_id})
    assert detail.status_code == 200
    assert detail.json()["status"] == "running"

    removal = client.post("/v1/job", json={"job_id": job_id, "rm": True})
    assert removal.status_code == 200
    assert removal.json()["status"] == "removed"

    missing = client.post("/v1/job", json={"job_id": job_id})
    assert missing.status_code == 400

    listing = client.post("/v1/job", json={})
    assert listing.status_code == 200
    assert set(listing.json().keys()) == set(job_ids[1:])


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("/v1/query/text_text", {"query": "hello"}),
        ("/v1/query/text_image", {"query": "hello"}),
        ("/v1/query/image_image", {"path": "/tmp/image.png"}),
        ("/v1/query/text_audio", {"query": "hello"}),
        ("/v1/query/audio_audio", {"path": "/tmp/audio.wav"}),
        ("/v1/query/text_video", {"query": "hello"}),
        ("/v1/query/image_video", {"path": "/tmp/image.png"}),
        ("/v1/query/audio_video", {"path": "/tmp/audio.wav"}),
        ("/v1/query/video_video", {"path": "/tmp/video.mp4"}),
    ],
)
def test_query_endpoints(api_client, endpoint, payload):
    client, _ = api_client

    resp = client.post(endpoint, json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["documents"]) == 2
