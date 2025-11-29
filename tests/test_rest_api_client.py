from __future__ import annotations

from tests.utils import mock_rest_api_client as mock


def test_client_smoke():
    for call in mock.CLIENT_CALLS:
        call()
