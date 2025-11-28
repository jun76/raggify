from __future__ import annotations

from raggify_client import RestAPIClient


def test_client_smoke():
    client = RestAPIClient()
    assert client.status is not None  # In practice, mock requests
