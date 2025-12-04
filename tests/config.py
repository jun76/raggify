from __future__ import annotations

import os
import warnings

import pytest
from pydantic import PydanticDeprecatedSince211
from pydantic.warnings import UnsupportedFieldAttributeWarning

__all__ = ["configure_test_env", "pytestmark"]


def configure_test_env() -> None:
    # The NodeParser on the llama_index side has validate_default=True set for the Annotated type,
    # and since this generates a warning every time pytest runs, we'll handle it in the test.
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

    # Suppressing the __get_pydantic_core_schema__ warning in Pydantic v2.11 and later
    warnings.filterwarnings("ignore", category=PydanticDeprecatedSince211)

    os.environ.setdefault("OPENAI_API_KEY", "dummy")
    os.environ.setdefault("COHERE_API_KEY", "dummy")
    os.environ.setdefault("VOYAGE_API_KEY", "dummy")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "dummy")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "dummy")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
    os.environ.setdefault("LLAMA_CLOUD_API_KEY", "dummy")
    os.environ.setdefault("RG_CONFIG_PATH", "tests/config.yaml")


pytestmark = pytest.mark.filterwarnings(
    "ignore::pydantic.warnings.PydanticDeprecatedSince211"
)
