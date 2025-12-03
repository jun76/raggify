from __future__ import annotations

from tests.utils import mock_rest_api_client as mock

from .config import configure_test_env

configure_test_env()


def test_client_smoke():
    for call in mock.CLIENT_CALLS:
        call()
